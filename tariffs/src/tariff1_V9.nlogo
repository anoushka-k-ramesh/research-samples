extensions [csv]
; this line is so that we can use .csv files

globals [
  us-current-inventory
  row-current-inventory
  us-looking
  row-looking
  previous-values
  no-change-ticks
  no-change-ticks-max
  us_tariffs
  row_tariffs
  us_incentives
  row_incentives
  dollars_domestic_sales_in_US
  dollars_foreign_sales_in_US
  dollars_domestic_sales_in_ROW
  dollars_foreign_sales_in_ROW
  count_domestic_sales_in_US
  count_foreign_sales_in_US
  count_domestic_sales_in_ROW
  count_foreign_sales_in_ROW
  total-wealth
  US-GINI
  ROW-GINI
  US-per-capita-GDP
  ROW-per-capita-GDP
  US-buyer-wealth
  ROW-buyer-wealth
  number-US-buyers
  number-ROW-buyers
]
; these lines create global variables
; us-current-inventory = remaining inventory of US EVs
; row-current-inventory = remaining inventory of foreign EVs
; us-looking = remaining US buyers looking to buy an EV
; row-looking = remaining foreign buyers looking to buy an EV
; previous-values, no-change-ticks, and no-change-ticks-max are related to stopping the simultion (otherwise, it would run an infinite loop)
; us_tariffs = total amount the US government made from tariffs on EVs
; row_tariffs = total amount the foreign government made from tariffs on EVs
; us_incentives = total amount the US government gave to US buyers to purchase domestic EVs
; row_incentives = total amount the foreign government gave to foreign buyers to purchase domestic EVs

;let index who
;setxy random-xcor random-ycor
;set color item 0 item index turtle-data
;set size item 1 item index turtle-data

breed [buyers buyer]
breed [sellers seller]
; breed creates two types of turtles (i.e. agents): buyers and sellers

buyers-own [money region looking purchase-price purchase-region incentive]
sellers-own [seller-type region current-inventory initial-inventory current-price sold-total-price min-price max-price tariff]
; buyers-own and sellers-own refer to individual agent-specific variables
; money = how much money the buyer has to  buy a car (numeric value from data csv upload)
; region = buyer's location (US or foreign); strings are "USA" and "ROW"
; looking = whether they have bought an EV or not (the only agents created in this game are the ones looking to buy an EV); Boolean
; purchase-price = if looking = false (i.e. they already bought an EV), what price did they buy the EV for? This is important to track since the price is dynamic
; purchase-region = if looking = false (i.e. they already bought an EV), did they buy "USA" or "ROW" (or eventually "USA"/"China"/"Europe")?
; incentive = increases the amount of money the buyer has only if they purchase a domestic EV; the incentive is esablished on the interface page via a user slider
; seller-type = manufacturer tier (i.e. luxury, standard, or budget) (so far, we only have done Tier 2/standard)
; region = US, China, or Europe
; current-inventory = how many EVs they have left to sell
; initial-inventory = set at 10 initially while we figure out how to finish the code; eventually, it will be from csv data upload
; current-price = what price they're selling the next EV at (need to still add tariff into code)
; sold-total-price = amount of total money manufacturer made
; min-price = the minimum price the manufacturer is willing to sell; based on a given range from a csv data upload
; max-price = the maximum price the manufacturer is willing to sell; based on a given range from a csv data upload
; tariff = the tariff imposed on imported cars (for the purposes of this model, uniform tariff around the world for all companies in the same tier)


to-report random-in-range [a b]
  report a + random (b + 1 - a)
end
; this is a function to be called on later -> it provides a random number within a specified range
; we need this function because the NetLogo version of a random number generated only takes one value

; this is a function to be called on later -> this is to randomly sample once from the Lorenz curve
to-report random-wealth [ GINI total-wealth-var count-buyers ]
  let wealth-share random-float 1
  let scaled-share (1 - GINI) * wealth-share + GINI * (wealth-share ^ 2)
  report scaled-share * total-wealth-var / count-buyers
end

to setup
  clear-all

  ask patches [
    set pcolor gray
    if pxcor >= 0 and pycor > 0 [
      set pcolor 139  ; Quadrant 1
    ]
    if pxcor <= 0 and pycor > 0 [
      set pcolor 139  ; Quadrant 2
    ]
    if pxcor <= 0 and pycor < 0 [
      set pcolor 59  ; Quadrant 3
    ]
    if pxcor >= 0 and pycor < 0 [
      set pcolor 59  ; Quadrant 4
    ]
  ]

  set dollars_domestic_sales_in_US 0
  set dollars_foreign_sales_in_US 0
  set dollars_domestic_sales_in_ROW 0
  set dollars_foreign_sales_in_ROW 0
  set count_domestic_sales_in_US 0
  set count_foreign_sales_in_US 0
  set count_domestic_sales_in_ROW 0
  set count_foreign_sales_in_ROW 0
  set total-wealth 81000 * 400  ; Assuming 400 agents for simplicity
  set US-GINI 0.41 ; this is US's 2023 GINI, according to https://ourworldindata.org/grapher/economic-inequality-gini-index?time=latest
  set US-per-capita-GDP 65020.35 ; this is US's 2023 per capita GDP, according to Trading Economics
  set ROW-GINI 0.36 ; this is China's 2023 GINI, according to https://ourworldindata.org/grapher/economic-inequality-gini-index?time=latest
  set ROW-per-capita-GDP 12174 ; this is China's 2023 per capita GDP, according to Trading Economics
  set number-US-buyers 1200000 / 100 ; scaled down by 100 (originally 1.2 million)
  set number-ROW-buyers 600000 / 100 ; scaled down by 100 (originally 600,000)
  set US-buyer-wealth number-US-buyers * US-per-capita-GDP
  set ROW-buyer-wealth number-ROW-buyers * US-per-capita-GDP

  ;let buyer-data-raw csv:from-file "/Users/ramesh/Documents/NetLogo/Data/sample_tariff_data_buyers.csv"
  ;let buyer-data-headers first buyer-data-raw
  ;let buyer-data but-first buyer-data-raw
  ;this skips the header row
  ;let seller-data-raw csv:from-file "/Users/ramesh/Documents/NetLogo/Data/sample_tariff_data_sellers.csv"
  ;let seller-data-headers first seller-data-raw
  ;let seller-data but-first seller-data-raw
  ;this skips the header row

  ; if you want to show a variable's data, use the show command: show buyer-data-raw

  ; Create 10 agents for the top half
  create-buyers number-ROW-buyers
  [
    let y-pos random-float (max-pycor - 1) + 1
    while [y-pos = 0] [ set y-pos random-float (max-pycor - 1) + 1 ]
    setxy random-xcor y-pos
    set region "ROW"
    set incentive foreign_buyer_incentive_slider
    set color blue
    set shape "person"
    set looking true
    set money random-wealth ROW-GINI ROW-buyer-wealth number-ROW-buyers
    set money money * percentage_of_wealth_allocated / 100
  ]
  ; Create 10 agents for the bottom half
  create-buyers number-US-buyers
  [
    let y-pos random-float (min-pycor + 1) - 1
    while [y-pos = 0] [ set y-pos random-float (min-pycor + 1) - 1 ]
    setxy random-xcor y-pos
    set region "USA"
    set incentive US_buyer_incentive_slider
    set color blue
    set shape "person"
    set looking true
    set money random-wealth US-GINI US-buyer-wealth number-US-buyers
    set money money * percentage_of_wealth_allocated / 100
  ]

  create-sellers 4
  [
    setxy random-xcor 0
    set shape "car"
  ]
  ; print sellers
  ; Unique actions based on agent ID
  ask sellers
  [
    ; print who
    if who = number-US-buyers + number-ROW-buyers ; ROW standard model
    [
      set region "ROW"
      set color red
      set seller-type "standard"
      set initial-inventory 100020 / 100; 100020 scaled down by 100
      set current-inventory initial-inventory
      set max-price 31336
      set min-price 0.95 * max-price
    ]
    if who = number-US-buyers + number-ROW-buyers + 1 ; ROW budget model
    [
      set region "ROW"
      set color red
      set seller-type "budget"
      set initial-inventory 127320 / 100 ; 127323 scaled down by 100
      set current-inventory initial-inventory
      set max-price 27400
      set min-price 0.97 * max-price
    ]
    if who = number-US-buyers + number-ROW-buyers + 2 ; US standard model
    [
      set region "USA"
      set color 63 ; sets to dark green
      set seller-type "standard"
      set initial-inventory 841980 / 100 ; 841982 scaled down by 100
      set current-inventory initial-inventory
      set max-price 46630
      set min-price 0.95 * max-price
    ]
    if who = number-US-buyers + number-ROW-buyers + 3 ; US budget model
    [
      set region "USA"
      set color 63 ; sets to dark green
      set seller-type "budget"
      set initial-inventory 375270 / 100 ; 375270 scaled down by 100
      set current-inventory initial-inventory
      set max-price 40630
      set min-price 0.97 * max-price
    ]
  ]

  ask buyers [ if money < (27400 * 0.97 - 10000 - 1) [ die ] ]

  ; reset-globals (don't need it since it's called below)
  update-globals

  set no-change-ticks-max 20
  set previous-values []
  set no-change-ticks 0
  ; these lines prevent an infinite loop

  reset-ticks ; sets ticks back to zero
end
; this connects back to the setup button on the Interface (essentially, restarts it)

to reset-globals
  set us-current-inventory 0
  set row-current-inventory 0
  set us-looking 0
  set row-looking 0
end

to update-globals
  reset-globals
  ask sellers [
    if region = "USA" [set us-current-inventory us-current-inventory + current-inventory]
    if region = "ROW" [set row-current-inventory row-current-inventory + current-inventory]
  ]
  ask buyers [
    if region = "USA" and looking [set us-looking us-looking + 1]
    if region = "ROW" and looking [set row-looking row-looking + 1]
  ]
end
; re-calculating how many EVs are left and how many buyers are still looking

to update-charts
  set-current-plot "Available Inventory"
  set-current-plot-pen "us-inventory"
  ; set-plot-pen-color green --> color actually needs to be set manually so this line is not needed
  set-current-plot-pen "row-inventory"
  ; set-plot-pen-color red

  set-current-plot "Remaining Buyers"
  set-current-plot-pen "us-buyers"
  ; set-plot-pen-color green
  set-current-plot-pen "row-buyers"
  ; set-plot-pen-color red
  set-current-plot "Revenues in American Market"
  set-current-plot-pen "us-revenue"
  set-current-plot-pen "chinese-revenue"
  set-current-plot "Revenues in Chinese Market"
  set-current-plot-pen "us-revenue"
  set-current-plot-pen "chinese-revenue"

end

to check-for-changes [current-values]
  ifelse current-values = previous-values [
    set no-change-ticks no-change-ticks + 1
  ] [
    set no-change-ticks 0
  ]
  set previous-values current-values
; this prevents infinite loops

end

; this is the beginning of tick running (it runs once per tick)
to go

  ;show previous-values
  ;show no-change-ticks

   if no-change-ticks >= no-change-ticks-max [
     stop
   ]

    ask buyers [
      if looking = true [
        let buyer-region-value region
        let available-sellers sellers with [current-inventory > 0]
        let available-domestic-sellers available-sellers with [region = buyer-region-value]
        let available-foreign-sellers available-sellers with [region != buyer-region-value]
        let optimal-seller nobody
        let optimal-domestic-seller min-one-of available-domestic-sellers [ current-price ] ; identifies cheapest domestic-seller the buyer could buy from in this tick
        let current-price-this-tick 0

        ; the following lines have each buyer identify the cheapest foreign-seller the buyer could buy from in this tick...
        ; and asks each foreign-seller who becomes a cheapest foreign-seller for any given buyer...
        ; to establish a current-price-this-tick variable and set it to the foreign-seller's asking price plus tariff or its max price, whichever is lower.

        let optimal-foreign-seller nobody

        if (count available-foreign-sellers > 0)
        [
          set optimal-foreign-seller min-one-of available-foreign-sellers [ current-price ]
          if (money > [current-price] of max-one-of available-foreign-sellers [ current-price ])
          [
            set optimal-foreign-seller max-one-of available-foreign-sellers [ current-price ]
          ]
        ]

        if (optimal-foreign-seller != nobody)
        [
          ask optimal-foreign-seller
          [
            ifelse (buyer-region-value = "USA")
            [
              ifelse (current-price * (1 + (US_tariff_slider_percent / 100)) <= max-price)
              [
                set current-price-this-tick (current-price * (1 + (US_tariff_slider_percent / 100))) ; 1st argument of the ifelse
                                                                                                     ;set current-price (current-price + tariff)
              ]
              [
                set current-price-this-tick max-price ; 2nd argument of the ifelse
                                                      ;set current-price max-price
              ]
            ]
            [
              ifelse (current-price * (1 + (foreign_tariff_slider_percent / 100)) <= max-price)
              [
                set current-price-this-tick (current-price * (1 + (foreign_tariff_slider_percent / 100)))
                ;set current-price (current-price + tariff)
              ]
              [
                set current-price-this-tick max-price
                ;set current-price max-price
              ]
            ]
          ]
        ]

        ; in the following lines, where a buyer faces an option between a domestic or foreign seller, the buyer...
        ; applies any incentive to the buyer's further comparison between foreign and domestic prices...
        ; and per the "ifelse" then sets either a seller or a price, accordingly

        if (optimal-domestic-seller != nobody and optimal-foreign-seller != nobody)
        [
          set optimal-seller optimal-domestic-seller ; the buyer sets that their optimal seller is the domestic seller
          ifelse ([current-price-this-tick] of optimal-foreign-seller < ([current-price] of optimal-domestic-seller - incentive)) ; buyer then compares the foreign price with the incentived domestic price
          [
            set optimal-seller optimal-foreign-seller
            ; if foreign price is still cheaper, buyer resets optimal-seller to be the foreign seller (1st argument of the ifelse)
          ]
          [
            set current-price-this-tick [ current-price] of optimal-domestic-seller
            ; if the domestic seller is now cheaper or is equal (2nd argument of the ifelse), ...
            ; buyer sets its local current-price-this-tick to be the domestic seller's current price without reference to the incentive; doesn't set the optimal buyer yet.
          ]
        ]
        ; if there is only a domestic seller, the buyer sets current price at the price of the domestic seller without reference to the incentive
        if (optimal-domestic-seller != nobody and optimal-foreign-seller = nobody)
        [
          set optimal-seller optimal-domestic-seller
          set current-price-this-tick [current-price] of optimal-domestic-seller
        ]
        ; if there is only a foreign seller, the buyer sets the seller as the optimal foreign seller
        if (optimal-foreign-seller != nobody and optimal-domestic-seller = nobody)
        [
          set optimal-seller optimal-foreign-seller
        ]
        ; If/else built on condition that there are no sellers...
        ; If there are no sellers, the program will print a message accordingly
        ifelse (optimal-foreign-seller = nobody and optimal-domestic-seller = nobody)
        [
          stop
          ; print "NO INVENTORY LEFT FOR EITHER COMPANY. ALL CARS SOLD." ; 1st argument of the ifelse
        ]
        [
          if (optimal-seller != nobody) ; the 2nd argument of the "ifelse" introduces an "if" that all of the other tests so far have already passed, so all buyers who found sellers move forward.
          [
            let all_money money + incentive ; this establishes that a buyer has extra buying power if they have an incentive
            ifelse all_money >= [current-price] of optimal-seller ; the following is the if portion of the ifelse loop
            [ ; the 1st argument of the "ifelse" -- what the buyer does if they can afford the car: reduce money remaining, stop looking, recolor self per origin of car
              set all_money all_money - [current-price] of optimal-seller
              set looking false
              if [region] of optimal-seller = "USA" [set color 63] ; sets to dark green
              if [region] of optimal-seller = "ROW" [set color red]
              ask optimal-seller [
                ; here is where the transaction has occurred
                ; establishes things the seller will do if a buyer can afford to buy their car: reduce inventory, log the sale to the total sales number, increase asking price for next round
                set current-inventory current-inventory - 1
                set sold-total-price sold-total-price + [current-price] of optimal-seller
                if ([region] of optimal-seller = "USA" and buyer-region-value = "USA")
                [
                  set dollars_domestic_sales_in_US (dollars_domestic_sales_in_US + sold-total-price)
                  set count_domestic_sales_in_US (count_domestic_sales_in_US + 1)
                ]
                if ([region] of optimal-seller = "ROW" and buyer-region-value = "ROW")
                [
                  set dollars_domestic_sales_in_ROW (dollars_domestic_sales_in_ROW + sold-total-price)
                  set count_domestic_sales_in_ROW (count_domestic_sales_in_ROW + 1)
                ]
                if ([region] of optimal-seller = "USA" and buyer-region-value = "ROW")
                [
                  set dollars_foreign_sales_in_ROW (dollars_foreign_sales_in_ROW + sold-total-price)
                  set count_foreign_sales_in_ROW (count_foreign_sales_in_ROW + 1)
                ]
                if ([region] of optimal-seller = "ROW" and buyer-region-value = "USA")
                [
                  set dollars_foreign_sales_in_US (dollars_foreign_sales_in_US + sold-total-price)
                  set count_foreign_sales_in_US (count_foreign_sales_in_US + 1)
                ]
                if current-inventory > 0 and current-price + company_price_increment < max-price [
                  set current-price current-price + company_price_increment
                ]
              ]
            ]
            ; the following is the else portion of the ifelse loop
            [
              ask available-sellers [
                if (current-price - company_price_increment) >= min-price [ ; while the successful seller increases their asking price, all other sellers reduce theirs
                  set current-price current-price - company_price_increment
                ]
              ]
            ]
          ]
        ]
      ]
    ]
    update-globals
    update-charts

  check-for-changes (list us-current-inventory row-current-inventory us-looking row-looking)

  tick



end
@#$#@#$#@
GRAPHICS-WINDOW
300
15
1112
668
-1
-1
4.0
1
10
1
1
1
0
0
0
1
-100
100
-80
80
1
1
1
ticks
30.0

BUTTON
15
75
81
108
setup
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
15
125
97
158
go-once
go
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

BUTTON
15
175
79
209
go
go
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

PLOT
1205
10
1405
160
Available Inventory
Tick
Inventory
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"us-inventory" 1.0 0 -13840069 true "" "plot us-current-inventory"
"row-inventory" 1.0 0 -2674135 true "" "plot row-current-inventory"

PLOT
1205
175
1405
325
Remaining Buyers
Tick
Buyers
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"us-buyers" 1.0 0 -11085214 true "" "plot us-looking"
"row-buyers" 1.0 0 -2674135 true "" "plot row-looking"

SLIDER
15
230
227
263
US_tariff_slider_percent
US_tariff_slider_percent
0
200
25.0
1
1
%
HORIZONTAL

SLIDER
15
350
272
383
US_buyer_incentive_slider
US_buyer_incentive_slider
0
10000
7500.0
500
1
USD
HORIZONTAL

SLIDER
15
290
267
323
foreign_tariff_slider_percent
foreign_tariff_slider_percent
0
200
0.0
1
1
NIL
HORIZONTAL

SLIDER
15
410
287
443
foreign_buyer_incentive_slider
foreign_buyer_incentive_slider
0
10000
7500.0
500
1
USD
HORIZONTAL

PLOT
1205
340
1405
490
Revenues in American Market
Tick
Price (USD)
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"us-revenue" 1.0 0 -13840069 true "" "plot dollars_domestic_sales_in_US"
"chinese-revenue" 1.0 0 -2674135 true "" "plot dollars_foreign_sales_in_US"

PLOT
1205
505
1405
655
Revenues in Chinese Market
Tick
Price (USD)
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"us-revenue" 1.0 0 -13840069 true "" "plot dollars_foreign_sales_in_ROW"
"chinese-revenue" 1.0 0 -2674135 true "" "plot dollars_domestic_sales_in_ROW"

SLIDER
15
470
257
503
company_price_increment
company_price_increment
0
1000
100.0
10
1
USD
HORIZONTAL

SLIDER
15
525
277
558
percentage_of_wealth_allocated
percentage_of_wealth_allocated
0
100
30.0
1
1
%
HORIZONTAL

@#$#@#$#@
## WHAT IS IT?

This model demonstrates a very simple bidding market, where buyers and sellers try to get the best price for goods in a competitive setting.

## HOW IT WORKS

The agents in the inner ring of represent sellers, who bring some goods they want to sell to the market. The agents in the outer ring are the buyers, who bring money to the market to buy the goods.

Each round (tick), the buyers move to their right (counter-clockwise) and are paired with a seller. Then each buyer checks its paired seller's asking price. If it's lower than their asking-price, they'll buy 1 item. If it's higher, they'll buy nothing. Then each seller individually decides to either raise prices or lower prices based on their buyer's behavior. The buyers also individually lower or raise their expectations based on their seller's behavior.

The market will run through rounds until no buyer who wants to buy an item has money left to spend, or until all items are bought.

### PURCHASE DISPLAY

If a purchase is made, the link from seller to buyer turns yellow and a little star animation will start. Otherwise, the link will be blue with no animation.  If the transactions still take place too quickly to be noticed, the speed slider can slow the model down a bit.

As buyers get what they want, they get bigger and fill their maximum-buy-sized gray shadow. If they fully satisfy their demand, they'll turn dark grey.

Sellers start with a size based on how many items they brought to the market.  As they sell items, they get smaller. If they sell all their items, they'll turn dark grey.

## HOW TO USE IT

Press SETUP to generate a market, then the GO-ONCE button to run a single round, or the GO button to run the market until it completes.

* Initial supply and demand can be High or Low, and distributed can be even among all buyers and sellers or concentrated among a few.
* Behavior is set when SETUP is run and can be set to one of a few options (the description given is for buyers, but mirrored for sellers):
    * Normal - buyers will increase their willing to pay by a small amount if they are unable to make a purchase, otherwise they'll lower their willing-to-pay if they do manage to buy an item.
    * Desperate - increases the amount buyers will increase thier willing-to-pay when they fail to make a purchase.
    * Random - the behavior of each buyer will be random - some will be very desperate, others might decrease the amount they're willing-to-pay when they fail to make a purchase.
    * Mix of all - each buyer will be set to one of the three above behaviors randomly.
* You can also toggle whether sellers will consider full buyers or not. A full buyer is one who has already satisfied their demand. This would simulate the sellers being able to tell who is walking past their market stall without even looking at the items' prices.

## THINGS TO NOTICE

Try slowing the tick speed down and watching a single buyer on the outside ring as it moves. See how it gets bigger when it makes a purchase and its link turns yellow, and how big it is when it turns completely dark grey (if it does).

Run the model multiple times with the same settings to get an idea of what's happening to the asking and buying prices over time.

## THINGS TO TRY

Play around with the different setup options to try the following:

* Can you get to 100% of both items sold and demand satisfied? If not, how close can you get?
* What's the longest you can get a market to run for (in number of ticks)?
* What's the highest average price you can get at the end of a round?
* Sometimes the market stops when there are buyers on the outside ring who still want to purchase things (% Demand Satisfied is not 100%) and there is still money available to spend (% Money Taken is not 100%). How this can be?

## EXTENDING THE MODEL

It's fairly easy to add new behaviors to buyers and sellers, just adjust the chooser box with a new option, then add it in the appropriate `create-ordered-sellers` or `create-ordered-buyers` block in the setup procedure.

## NETLOGO FEATURES

In order to try to keep the Asking Price and Buying Price plots displaying relevant information, we set the `plot-y-range` manually using an update command based on the most recent average price.

Behaviors for buyers and sellers are set during setup, and we use anonymous procedures stored in turtles-own variables. This allows us to very easily execute different behavior while the market is running by just using the `run` keyword with those behavior variables.

## RELATED MODELS

See the Simple Economy model or Sugarscape models to explore other economic concepts.

## HOW TO CITE

If you mention this model or the NetLogo software in a publication, we ask that you include the citations below.

For the model itself:

* Baker, J. and Wilensky, U. (2017).  NetLogo Bidding Market model.  http://ccl.northwestern.edu/netlogo/models/BiddingMarket.  Center for Connected Learning and Computer-Based Modeling, Northwestern University, Evanston, IL.

Please cite the NetLogo software as:

* Wilensky, U. (1999). NetLogo. http://ccl.northwestern.edu/netlogo/. Center for Connected Learning and Computer-Based Modeling, Northwestern University, Evanston, IL.

## COPYRIGHT AND LICENSE

Copyright 2017 Uri Wilensky.

![CC BY-NC-SA 3.0](http://ccl.northwestern.edu/images/creativecommons/byncsa.png)

This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.  To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/3.0/ or send a letter to Creative Commons, 559 Nathan Abbott Way, Stanford, California 94305, USA.

Commercial licenses are also available. To inquire about commercial licenses, please contact Uri Wilensky at uri@northwestern.edu.

<!-- 2017 Cite: Baker, J. -->
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

airplane
true
0
Polygon -7500403 true true 150 0 135 15 120 60 120 105 15 165 15 195 120 180 135 240 105 270 120 285 150 270 180 285 210 270 165 240 180 180 285 195 285 165 180 105 180 60 165 15

arrow
true
0
Polygon -7500403 true true 150 0 0 150 105 150 105 293 195 293 195 150 300 150

box
false
0
Polygon -7500403 true true 150 285 285 225 285 75 150 135
Polygon -7500403 true true 150 135 15 75 150 15 285 75
Polygon -7500403 true true 15 75 15 225 150 285 150 135
Line -16777216 false 150 285 150 135
Line -16777216 false 150 135 15 75
Line -16777216 false 150 135 285 75

bug
true
0
Circle -7500403 true true 96 182 108
Circle -7500403 true true 110 127 80
Circle -7500403 true true 110 75 80
Line -7500403 true 150 100 80 30
Line -7500403 true 150 100 220 30

butterfly
true
0
Polygon -7500403 true true 150 165 209 199 225 225 225 255 195 270 165 255 150 240
Polygon -7500403 true true 150 165 89 198 75 225 75 255 105 270 135 255 150 240
Polygon -7500403 true true 139 148 100 105 55 90 25 90 10 105 10 135 25 180 40 195 85 194 139 163
Polygon -7500403 true true 162 150 200 105 245 90 275 90 290 105 290 135 275 180 260 195 215 195 162 165
Polygon -16777216 true false 150 255 135 225 120 150 135 120 150 105 165 120 180 150 165 225
Circle -16777216 true false 135 90 30
Line -16777216 false 150 105 195 60
Line -16777216 false 150 105 105 60

car
false
0
Polygon -7500403 true true 300 180 279 164 261 144 240 135 226 132 213 106 203 84 185 63 159 50 135 50 75 60 0 150 0 165 0 225 300 225 300 180
Circle -16777216 true false 180 180 90
Circle -16777216 true false 30 180 90
Polygon -16777216 true false 162 80 132 78 134 135 209 135 194 105 189 96 180 89
Circle -7500403 true true 47 195 58
Circle -7500403 true true 195 195 58

circle
false
0
Circle -7500403 true true 0 0 300

circle 2
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240

cow
false
0
Polygon -7500403 true true 200 193 197 249 179 249 177 196 166 187 140 189 93 191 78 179 72 211 49 209 48 181 37 149 25 120 25 89 45 72 103 84 179 75 198 76 252 64 272 81 293 103 285 121 255 121 242 118 224 167
Polygon -7500403 true true 73 210 86 251 62 249 48 208
Polygon -7500403 true true 25 114 16 195 9 204 23 213 25 200 39 123

cylinder
false
0
Circle -7500403 true true 0 0 300

dot
false
0
Circle -7500403 true true 90 90 120

face happy
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 255 90 239 62 213 47 191 67 179 90 203 109 218 150 225 192 218 210 203 227 181 251 194 236 217 212 240

face neutral
false
0
Circle -7500403 true true 8 7 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Rectangle -16777216 true false 60 195 240 225

face sad
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 168 90 184 62 210 47 232 67 244 90 220 109 205 150 198 192 205 210 220 227 242 251 229 236 206 212 183

fish
false
0
Polygon -1 true false 44 131 21 87 15 86 0 120 15 150 0 180 13 214 20 212 45 166
Polygon -1 true false 135 195 119 235 95 218 76 210 46 204 60 165
Polygon -1 true false 75 45 83 77 71 103 86 114 166 78 135 60
Polygon -7500403 true true 30 136 151 77 226 81 280 119 292 146 292 160 287 170 270 195 195 210 151 212 30 166
Circle -16777216 true false 215 106 30

flag
false
0
Rectangle -7500403 true true 60 15 75 300
Polygon -7500403 true true 90 150 270 90 90 30
Line -7500403 true 75 135 90 135
Line -7500403 true 75 45 90 45

flower
false
0
Polygon -10899396 true false 135 120 165 165 180 210 180 240 150 300 165 300 195 240 195 195 165 135
Circle -7500403 true true 85 132 38
Circle -7500403 true true 130 147 38
Circle -7500403 true true 192 85 38
Circle -7500403 true true 85 40 38
Circle -7500403 true true 177 40 38
Circle -7500403 true true 177 132 38
Circle -7500403 true true 70 85 38
Circle -7500403 true true 130 25 38
Circle -7500403 true true 96 51 108
Circle -16777216 true false 113 68 74
Polygon -10899396 true false 189 233 219 188 249 173 279 188 234 218
Polygon -10899396 true false 180 255 150 210 105 210 75 240 135 240

house
false
0
Rectangle -7500403 true true 45 120 255 285
Rectangle -16777216 true false 120 210 180 285
Polygon -7500403 true true 15 120 150 15 285 120
Line -16777216 false 30 120 270 120

leaf
false
0
Polygon -7500403 true true 150 210 135 195 120 210 60 210 30 195 60 180 60 165 15 135 30 120 15 105 40 104 45 90 60 90 90 105 105 120 120 120 105 60 120 60 135 30 150 15 165 30 180 60 195 60 180 120 195 120 210 105 240 90 255 90 263 104 285 105 270 120 285 135 240 165 240 180 270 195 240 210 180 210 165 195
Polygon -7500403 true true 135 195 135 240 120 255 105 255 105 285 135 285 165 240 165 195

line
true
0
Line -7500403 true 150 0 150 300

line half
true
0
Line -7500403 true 150 0 150 150

pentagon
false
0
Polygon -7500403 true true 150 15 15 120 60 285 240 285 285 120

person
false
0
Circle -7500403 true true 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105

plant
false
0
Rectangle -7500403 true true 135 90 165 300
Polygon -7500403 true true 135 255 90 210 45 195 75 255 135 285
Polygon -7500403 true true 165 255 210 210 255 195 225 255 165 285
Polygon -7500403 true true 135 180 90 135 45 120 75 180 135 210
Polygon -7500403 true true 165 180 165 210 225 180 255 120 210 135
Polygon -7500403 true true 135 105 90 60 45 45 75 105 135 135
Polygon -7500403 true true 165 105 165 135 225 105 255 45 210 60
Polygon -7500403 true true 135 90 120 45 150 15 180 45 165 90

sheep
false
15
Circle -1 true true 203 65 88
Circle -1 true true 70 65 162
Circle -1 true true 150 105 120
Polygon -7500403 true false 218 120 240 165 255 165 278 120
Circle -7500403 true false 214 72 67
Rectangle -1 true true 164 223 179 298
Polygon -1 true true 45 285 30 285 30 240 15 195 45 210
Circle -1 true true 3 83 150
Rectangle -1 true true 65 221 80 296
Polygon -1 true true 195 285 210 285 210 240 240 210 195 210
Polygon -7500403 true false 276 85 285 105 302 99 294 83
Polygon -7500403 true false 219 85 210 105 193 99 201 83

square
false
0
Rectangle -7500403 true true 30 30 270 270

square 2
false
0
Rectangle -7500403 true true 30 30 270 270
Rectangle -16777216 true false 60 60 240 240

star
false
0
Polygon -7500403 true true 151 1 185 108 298 108 207 175 242 282 151 216 59 282 94 175 3 108 116 108

target
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240
Circle -7500403 true true 60 60 180
Circle -16777216 true false 90 90 120
Circle -7500403 true true 120 120 60

tree
false
0
Circle -7500403 true true 118 3 94
Rectangle -6459832 true false 120 195 180 300
Circle -7500403 true true 65 21 108
Circle -7500403 true true 116 41 127
Circle -7500403 true true 45 90 120
Circle -7500403 true true 104 74 152

triangle
false
0
Polygon -7500403 true true 150 30 15 255 285 255

triangle 2
false
0
Polygon -7500403 true true 150 30 15 255 285 255
Polygon -16777216 true false 151 99 225 223 75 224

truck
false
0
Rectangle -7500403 true true 4 45 195 187
Polygon -7500403 true true 296 193 296 150 259 134 244 104 208 104 207 194
Rectangle -1 true false 195 60 195 105
Polygon -16777216 true false 238 112 252 141 219 141 218 112
Circle -16777216 true false 234 174 42
Rectangle -7500403 true true 181 185 214 194
Circle -16777216 true false 144 174 42
Circle -16777216 true false 24 174 42
Circle -7500403 false true 24 174 42
Circle -7500403 false true 144 174 42
Circle -7500403 false true 234 174 42

turtle
true
0
Polygon -10899396 true false 215 204 240 233 246 254 228 266 215 252 193 210
Polygon -10899396 true false 195 90 225 75 245 75 260 89 269 108 261 124 240 105 225 105 210 105
Polygon -10899396 true false 105 90 75 75 55 75 40 89 31 108 39 124 60 105 75 105 90 105
Polygon -10899396 true false 132 85 134 64 107 51 108 17 150 2 192 18 192 52 169 65 172 87
Polygon -10899396 true false 85 204 60 233 54 254 72 266 85 252 107 210
Polygon -7500403 true true 119 75 179 75 209 101 224 135 220 225 175 261 128 261 81 224 74 135 88 99

wheel
false
0
Circle -7500403 true true 3 3 294
Circle -16777216 true false 30 30 240
Line -7500403 true 150 285 150 15
Line -7500403 true 15 150 285 150
Circle -7500403 true true 120 120 60
Line -7500403 true 216 40 79 269
Line -7500403 true 40 84 269 221
Line -7500403 true 40 216 269 79
Line -7500403 true 84 40 221 269

wolf
false
0
Polygon -16777216 true false 253 133 245 131 245 133
Polygon -7500403 true true 2 194 13 197 30 191 38 193 38 205 20 226 20 257 27 265 38 266 40 260 31 253 31 230 60 206 68 198 75 209 66 228 65 243 82 261 84 268 100 267 103 261 77 239 79 231 100 207 98 196 119 201 143 202 160 195 166 210 172 213 173 238 167 251 160 248 154 265 169 264 178 247 186 240 198 260 200 271 217 271 219 262 207 258 195 230 192 198 210 184 227 164 242 144 259 145 284 151 277 141 293 140 299 134 297 127 273 119 270 105
Polygon -7500403 true true -1 195 14 180 36 166 40 153 53 140 82 131 134 133 159 126 188 115 227 108 236 102 238 98 268 86 269 92 281 87 269 103 269 113

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.4.0
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
<experiments>
  <experiment name="Tariff_Experiment" repetitions="1" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <metric>dollars_domestic_sales_in_US / (dollars_domestic_sales_in_US + dollars_foreign_sales_in_US)</metric>
    <steppedValueSet variable="US_tariff_slider_percent" first="0" step="5" last="100"/>
  </experiment>
  <experiment name="Buyer_Incentive_Experiment" repetitions="1" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <metric>dollars_domestic_sales_in_US / (dollars_domestic_sales_in_US + dollars_foreign_sales_in_US)</metric>
    <steppedValueSet variable="US_buyer_incentive_slider" first="0" step="500" last="10000"/>
  </experiment>
  <experiment name="Tariff_AND_Buyer_Incentive_Experiment_V1" repetitions="1" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <metric>dollars_domestic_sales_in_US / (dollars_domestic_sales_in_US + dollars_foreign_sales_in_US)</metric>
    <steppedValueSet variable="US_tariff_slider_percent" first="0" step="5" last="100"/>
    <steppedValueSet variable="US_buyer_incentive_slider" first="0" step="500" last="10000"/>
  </experiment>
  <experiment name="Tariff_AND_Buyer_Incentive_Experiment_V2" repetitions="1" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <metric>dollars_domestic_sales_in_US / (dollars_domestic_sales_in_US + dollars_foreign_sales_in_US)</metric>
    <steppedValueSet variable="US_tariff_slider_percent" first="0" step="5" last="100"/>
    <steppedValueSet variable="US_buyer_incentive_slider" first="0" step="2000" last="10000"/>
  </experiment>
</experiments>
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180

line
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 0 135 45
Line -7500403 true 150 0 165 45
@#$#@#$#@
1
@#$#@#$#@
