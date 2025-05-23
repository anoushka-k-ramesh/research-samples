from utilities import *
from model import *

import torch
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
import networkx as nx

def train(model, data, labels, hidden, optimizer):
    model.train()
    optimizer.zero_grad()
    edge_pred, hidden, _, _ = model(data.x, data.edge_index, data.edge_features, hidden)
    #edge_pred = torch.sum(out, dim=1)
    #print(edge_pred.shape, labels.shape)
    loss = F.binary_cross_entropy_with_logits(edge_pred.squeeze(-1), labels.float())
    loss.backward()
    optimizer.step()
    hidden = (hidden[0].detach(), hidden[1].detach())  # Detach hidden states
    return loss.item(), hidden

def train_all_models(train_years, model_folder, all_data, all_parms):
    assert min(train_years) > 1816 and max(train_years) <= 2012 
    for y in sorted(train_years):
        train_one_model(y, y-1, model_folder, all_data, all_parms)

def train_one_model(prediction_target_year, prediction_prior_year, model_folder, all_data, all_parms):
    model_name = 'base_' + str(prediction_target_year) + '.pt'
    if os.path.exists(model_folder + model_name):
        print('Model already trained, skipping', model_name)
        return

    model = GCN_LSTM(all_parms['in_channels'], all_parms['hidden_channels'], 
                     all_parms['out_channels'], all_parms['num_edge_features'],
                     num_layers=all_parms['num_layers'], dropout=all_parms['dropout'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=all_parms['learning_rate'], weight_decay=all_parms['weight_decay'])

    years = all_data['years']
    for epoch in range(1, all_parms['n_epochs']+1):
        hidden = None
        for i in range(years.index(prediction_target_year)):
            target_year = years[i]
            prior_year = years[i-1] if i > 0 else None
            target_edge_index = all_data['temporal_edges_tensor'][target_year] #list of node pairs (source to destination so likely repeated flipped)
            prior_edge_index = all_data['temporal_edges_tensor'][prior_year] if prior_year else None
            target_edge_features = all_data['temporal_edge_features_tensor'][target_year]       
            x_base = torch.tensor(list(all_data['yearly_base_features'][target_year].values()), dtype=torch.float32)
            x_regime = torch.tensor(list(all_data['yearly_regime_features'][target_year].values()), dtype=torch.float32)
            x_religion = torch.tensor(list(all_data['yearly_religion_features'][target_year].values()), dtype=torch.float32)
            x = torch.hstack([x_base, x_regime, x_religion])
            all_edges, all_edge_features, labels = create_samples(target_edge_index, target_edge_features, 
                                all_data['num_nodes'], negative_ratio = all_parms['negative_ratio'],
                                max_distance=all_parms['max_distance'], prior_edge_index=prior_edge_index)
            
            data = Data(x=x, edge_index=all_edges, edge_features=all_edge_features)
            loss, hidden = train(model, data, labels, hidden, optimizer)
        if epoch % all_parms['print_interval'] == 0:
            print (model_name, epoch, loss)
    torch.save(
        {'epoch': epoch, 'model_state_dict': model.state_dict(),
             'loss': float(loss), 'hidden_state': hidden
        }, model_folder + model_name)

def load_saved_model(model_folder, prediction_target_year, all_parms, restore=True):
    model = GCN_LSTM(all_parms['in_channels'], all_parms['hidden_channels'], 
                     all_parms['out_channels'], all_parms['num_edge_features'],
                     num_layers=all_parms['num_layers'], dropout=all_parms['dropout'])
    model_name = 'base_' + str(prediction_target_year) + '.pt'
    hidden = None
    if restore and os.path.exists(model_folder + model_name):
        checkpoint = torch.load(model_folder + model_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        hidden = checkpoint['hidden_state']
    return model, hidden

def get_all_evaluations(model_folder, eval_years, all_data, all_parms, restore=False, max_trained_year=2012):
    results_df = None
    for eval_year in eval_years:
        if eval_year <= max_trained_year:
            model_name = 'base_' + str(eval_year) + '.pt'
            model, hidden = load_saved_model(model_folder, eval_year, all_parms=all_parms, restore=restore)
            #if hidden is not None or not restore:   #model has been trained and saved in a checkpoint
            th, test_results, test_all_edges, test_edge_pred, test_pred, \
                test_labels, test_data, test_src, test_dest = \
                predict_test_year(model, eval_year, all_data, all_parms, hidden, eval_year-1)
            results_df = pd.DataFrame([test_results]) if results_df is None else results_df.append([test_results])

    return results_df

def predict_test_year(model, test_year, all_data, all_parms, hidden, prior_year):
    num_nodes = all_data['num_nodes']
    data = Data(x=None, edge_index=None, edge_features=None)
    edge_index = all_data['temporal_edges_tensor'][test_year]
    edge_features = all_data['temporal_edge_features_tensor'][test_year]
    all_edges, all_edge_features, labels = create_samples(edge_index, edge_features, 
                   num_nodes, negative_ratio = all_parms['negative_ratio'], 
                   max_distance=all_parms['max_distance'], prior_edge_index=all_data['temporal_edges_tensor'][prior_year])
    data.edge_index = all_edges
    data.edge_features = all_edge_features
    x_base = torch.tensor(list(all_data['yearly_base_features'][test_year].values()), dtype=torch.float32)
    x_regime = torch.tensor(list(all_data['yearly_regime_features'][test_year].values()), dtype=torch.float32)
    x_religion = torch.tensor(list(all_data['yearly_religion_features'][test_year].values()), dtype=torch.float32)
    data.x = torch.hstack([x_base, x_regime, x_religion])
    th, accuracy, precision, recall, f1, auc_pr, edge_pred, pred, labels, src, dest = test(model, data, labels, hidden)
    results= {
            'Year': test_year,
            'threshold': th,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'AUC-PR': auc_pr,
            #'TOPK-ACC': top_k_acc,
        }
    return th, results, all_edges, edge_pred, pred, labels, data, src, dest

def test(model, data, labels, hidden):
    model.eval()
    edge_pred, hidden, src, dest = model(data.x, data.edge_index, data.edge_features, hidden)    
    edge_pred = torch.sigmoid(edge_pred.squeeze(-1)).cpu().detach().numpy()
    trues = labels.detach().cpu().numpy()    
    precision, recall, f1, auc_pr, optimal_threshold, pred = find_optimal_threshold(trues, edge_pred)
    accuracy = sum(np.where(pred == trues, 1, 0)) / len(trues)
    return optimal_threshold, accuracy, precision, recall, f1, auc_pr, edge_pred, pred, trues, src, dest


def find_optimal_threshold(trues, edge_pred):
    precision_vals, recall_vals, pr_thresholds = precision_recall_curve(trues, edge_pred)
    epsilon = 1e-10
    f1_scores = 2 * precision_vals * recall_vals / (precision_vals + recall_vals + epsilon)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = pr_thresholds[optimal_idx]
    pred = np.where(edge_pred > optimal_threshold, 1, 0)
    precision = precision_vals[optimal_idx]
    recall = recall_vals[optimal_idx]
    f1 = f1_scores[optimal_idx]
    auc_pr = auc(recall_vals, precision_vals)
    if np.isnan(f1):
        f1 = 0.0
        precision = 0.0
        recall = 0.0    
    return precision, recall, f1, auc_pr, optimal_threshold, pred


def find_new_and_deleted_edges(edge_index_1, edge_index_2):
    # Convert edge tensors to sets of tuples for easy comparison
    edge_set_1 = set(map(tuple, edge_index_1.t().tolist()))
    edge_set_2 = set(map(tuple, edge_index_2.t().tolist()))
    
    # Find new edges in the second set that were not in the first set
    new_edges = edge_set_2 - edge_set_1
    
    # Find edges in the first set that were deleted in the second set
    deleted_edges = edge_set_1 - edge_set_2
    
    # Convert sets of tuples back to torch tensors
    new_edges_tensor = torch.tensor(list(new_edges), dtype=torch.long).t()
    deleted_edges_tensor = torch.tensor(list(deleted_edges), dtype=torch.long).t()
    
    return new_edges_tensor, deleted_edges_tensor

def find_unique_undirected_edges_indices(edges):
    # Convert to list of frozensets to ensure undirected uniqueness
    edge_list = [frozenset(edge) for edge in edges.t().tolist()]
    
    # Use dictionary to track first occurrence index of each unique edge
    unique_edge_indices = {}
    
    for index, edge in enumerate(edge_list):
        if edge not in unique_edge_indices:
            unique_edge_indices[edge] = index
    
    # Extract the indices of unique edges
    unique_indices = list(unique_edge_indices.values())
    
    return unique_indices

def calculate_edge_stats(df):
    total_edges_by_year = []
    new_edges_by_year = []
    deleted_edges_by_year = []
    years = sorted(df['year'].unique())

    previous_edges = set(df[df['year'] == years[0]][['ccode1', 'ccode2']].apply(tuple, axis=1))

    for year in years[1:]:
        current_edges = set(df[df['year'] == year][['ccode1', 'ccode2']].apply(tuple, axis=1))

        # Calculate total number of edges
        total_edges = len(current_edges)
        total_edges_by_year.append(total_edges)

        # Calculate new edges
        new_edges = current_edges - previous_edges
        new_edges_count = len(new_edges)
        new_edges_by_year.append(new_edges_count)

        # Calculate deleted edges
        deleted_edges = previous_edges - current_edges
        deleted_edges_count = len(deleted_edges)
        #print(year, total_edges, new_edges_count, deleted_edges_count)
        deleted_edges_by_year.append(deleted_edges_count)
        
        # Update previous edges for the next iteration
        previous_edges = current_edges

    # Create a results DataFrame
    results_df = pd.DataFrame({
        'year': years[1:],
        'total_edges': total_edges_by_year,
        'new_edges': new_edges_by_year,
        'deleted_edges': deleted_edges_by_year
    })

    return results_df

#Note we do not need frozen set here, order matters here 
def merge_edges_and_labels(edge_index_1, edge_index_features_1, labels_1, edge_index_2, labels_2):
    combined_edges = torch.cat([edge_index_1, edge_index_2], dim=1)
    combined_labels = torch.cat([labels_1, labels_2], dim=0)
    edge_list = [tuple(edge) for edge in combined_edges.t().tolist()]
    unique_edge_set = []
    unique_indices = []
    for idx, edge in enumerate(edge_list):
        if edge not in unique_edge_set:
            unique_edge_set.append(edge)
            unique_indices.append(idx)
    
    unique_edges = combined_edges[:, unique_indices]
    unique_labels = combined_labels[unique_indices]    
    return unique_edges, unique_labels

def merge_prior_year_negative_samples(prior_edge_index, current_edge_index, current_labels):
    _, deleted_edges = find_new_and_deleted_edges(prior_edge_index, current_edge_index)
    if deleted_edges is None or len(deleted_edges.shape) <= 1:
        return current_edge_index, current_labels
    
    deleted_labels = np.zeros(deleted_edges.shape[1])
    return merge_edges_and_labels(current_edge_index, current_labels, deleted_edges, deleted_labels)

def create_geodesic_negative_samples(edge_index, num_nodes, num_neg_samples=None, max_distance=3):
    if num_neg_samples is None:
        num_neg_samples = edge_index.size(1)

    G = nx.Graph()
    edges = edge_index.t().tolist()
    G.add_edges_from(edges)
    
    # Calculate shortest path lengths for all pairs of nodes
    path_lengths = dict(nx.all_pairs_shortest_path_length(G))
    
    nodes = list(G.nodes())
    node_pairs = [(u, v) for u in range(num_nodes) for v in range(num_nodes) if u != v and not G.has_edge(u, v)]
    
    # Filter node pairs based on geodesic distance
    filtered_pairs = []
    for u, v in node_pairs:
        distance = path_lengths.get(u, {}).get(v, float('inf'))
        if distance <= max_distance:
            filtered_pairs.append((u, v))
    
    # Sample negative edges based on geodesic distance
    neg_samples = random.sample(filtered_pairs, min(len(filtered_pairs), num_neg_samples))
    
    return torch.tensor(neg_samples, dtype=torch.long).t().contiguous()

# Function to create positive and hard negative samples
def create_hard_neg_samples(edge_index, num_nodes, num_neg_samples):
    pos_samples = edge_index.t().cpu().numpy()
    
    all_pairs = set((i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j)
    pos_pairs = set((u, v) for u, v in pos_samples)
    
    neg_pairs = list(all_pairs - pos_pairs)
    
    # Randomly shuffle and select num_neg_samples negative pairs
    random.shuffle(neg_pairs)
    neg_pairs = neg_pairs[:num_neg_samples]
    
    # Convert negative pairs to tensor
    neg_samples = torch.tensor(neg_pairs, dtype=torch.long).t().contiguous()
    
    return neg_samples

# Update the create_samples function to include hard negative mining
def create_samples(edge_index, edge_features, num_nodes, negative_ratio = 1, max_distance=3, prior_edge_index=None):
    pos_samples = edge_index
    pos_edge_features = edge_features
    num_neg_samples = edge_index.size(1) * negative_ratio  # Generate more negative samples
    
    #slightly harder mining
    #neg_samples = create_geodesic_negative_samples(edge_index, num_nodes, num_neg_samples=num_neg_samples, max_distance=max_distance)
    
    #hard mining
    neg_samples = create_hard_neg_samples(edge_index, num_nodes, num_neg_samples)
        
    #easy mining
    #neg_samples = negative_sampling(edge_index, num_nodes=num_nodes, num_neg_samples=num_neg_samples)
    
    #additional negative samples as deletes from prior year
    prior_delete_negs_flag = False
    if prior_edge_index is not None:
        _, prior_delete_negs = find_new_and_deleted_edges(prior_edge_index, edge_index)
        if len(prior_delete_negs) > 0:
            prior_delete_negs_flag = True
            neg_samples = torch.concatenate([neg_samples, prior_delete_negs], dim=1)
            num_neg_samples = num_neg_samples + prior_delete_negs.shape[1]
    
    y_pos = torch.ones(pos_samples.size(1), dtype=torch.long)
    if len(neg_samples.shape) > 1:
        y_neg = torch.zeros(neg_samples.size(1), dtype=torch.long)
        all_edges = torch.cat([pos_samples, neg_samples], dim=1)
        
        #Approach 1 set the negative edges to zero, too easy?
        #neg_edge_features = torch.zeros_like(edge_features[:num_neg_samples])
        #Approach 2 set it to random with similar distribution as the positives 
        mean = pos_edge_features.mean(dim=0)
        std = pos_edge_features.std(dim=0)
        neg_edge_features = torch.normal(mean=mean.unsqueeze(0).expand(num_neg_samples, -1), 
                                              std=std.unsqueeze(0).expand(num_neg_samples, -1))
        #Approach 3 -- data driven?
        
        all_edge_features = torch.cat([pos_edge_features, neg_edge_features], dim=0)
        labels = torch.cat([y_pos, y_neg], dim=0)
    else:
        all_edges = torch.cat([pos_samples], dim=1)
        all_edge_features = torch.cat([pos_edge_features], dim=0)
        labels = torch.cat([y_pos], dim=0)
    #print(pos_samples.shape, neg_samples.shape, all_edges.shape, all_edge_features.shape, labels.shape, prior_edge_index is not None, prior_delete_negs_flag)
    return all_edges, all_edge_features, labels

def visualize_embeddings(node_list, reduced_embeddings, actual_edge_index, predicted_edge_index, \
                         pos_labels, node_colors, title='Node Embeddings'):

    fig, axes = plt.subplots(2, 1, figsize=(20, 12))    
    
    axes[0].set_title('Observed Alliances')
    visible = np.zeros(len(node_list), dtype=bool)
    # Plot actual positive edges
    for i, (src, dst) in enumerate(actual_edge_index.t().cpu().numpy()):
        s = node_list.index(src)
        d = node_list.index(dst)
        visible[s] = True
        visible[d] = True
        axes[0].plot([reduced_embeddings[s, 0], reduced_embeddings[d, 0]], 
                [reduced_embeddings[s, 1], reduced_embeddings[d, 1]], 
                'lightgray', alpha=0.5)
        axes[0].set_xticklabels([])
        axes[0].set_yticklabels([])
    axes[0].scatter(reduced_embeddings[visible][:, 0], reduced_embeddings[visible][:, 1], \
                    c=np.array(node_colors)[visible])
    # Plot predicted positive edges
    axes[1].set_title('Predicted Alliances')
    
    visible = np.zeros(len(node_list), dtype=bool)
    for i, (src, dst) in enumerate(predicted_edge_index.t().cpu().numpy()):
        s = node_list.index(src)
        d = node_list.index(dst)
        visible[s] = True
        visible[d] = True
        axes[1].plot([reduced_embeddings[s, 0], reduced_embeddings[d, 0]], 
                 [reduced_embeddings[s, 1], reduced_embeddings[d, 1]], 
                 'skyblue', linestyle='-', alpha=0.5)
        axes[1].set_xticklabels([])
        axes[1].set_yticklabels([])
    
    axes[1].scatter(reduced_embeddings[visible][:, 0], reduced_embeddings[visible][:, 1], \
                    c=np.array(node_colors)[visible])
    plt.suptitle(title)
    plt.show()

def compare_model_to_data(eval_year, threshold, perplexity, model_folder, all_data, all_parms):
    model_name = 'base_' + str(eval_year) + '.pt'
    #model_folder = './models/'
    model, hidden = load_saved_model(model_folder, eval_year, all_parms=all_parms, restore=True)
    if hidden is not None:   #model has been trained and saved in a checkpoint
        th, test_results, test_all_edges, test_edge_pred, test_pred, \
            test_labels, test_data, test_src, test_dest = \
            predict_test_year(model, eval_year, all_data, all_parms, hidden, eval_year-1)

        actual_test_pos_edges = test_all_edges[:, test_labels == 1]
        pos_nodes = actual_test_pos_edges.flatten().unique().numpy()
        all_nodes = test_all_edges.flatten().unique().numpy()
        predicted_test_pos_edges = test_all_edges[:, (test_edge_pred > threshold).nonzero()[0]]

        embeddings = OrderedDict()
        for i in range(len(test_all_edges[0,:])):
            s, d = test_all_edges[:,i]
            s = int(s)
            d = int(d)
            if s not in embeddings:
                embeddings[s] = test_src[i].detach().cpu().numpy()
            if d not in embeddings:
                embeddings[d] = test_dest[i].detach().cpu().numpy()

        node_list = list(embeddings.keys())
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        reduced_embeddings = tsne.fit_transform(list(embeddings.values()))

        node_colors = ['red' for _ in range(len(node_list))]
        for n in node_list:
            if n in pos_nodes:
                node_colors[node_list.index(n)] = 'orange'

        title='Year ' + str(eval_year) + ' -- Comparing Model Predictions and Historical Data\nCount of known alliances=' + \
            str(len(actual_test_pos_edges[0,:])) + ', threshold = ' + str(threshold) + ', perplexity = ' + str(perplexity)
        visualize_embeddings(node_list, reduced_embeddings, actual_test_pos_edges, \
                             predicted_test_pos_edges, test_labels, node_colors, title=title)
