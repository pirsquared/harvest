Harvest Simulation's documentation!
===================================

What's inside?
--------------

This is a library to simulate various tax lot harvesting algorithms.

It's broken up into three modules

1. Account.py
   Contains the class definition for the account object.  This acts as a light weight accounting system.
   Contains the main entry function to execute trading simulations.
   Methods that specify the trading algorithms
2. simutil.py
   Functions for producing simulation data and other simulation processing.
3. lotutil.py
   Functions for handling tax lot manipulation.
   Contains specifications for various tax lot release methodologies.

What's the purpose?
-------------------

I will use the trading algorithms to evaluate gains and losses as well as ability to track changing allocation
over a simulation period of approximately 10 years.

Algorithm Definitions
=====================

Allocation
----------

This is the most naive of the algorithms.  We simply calculate the required trades to get
to our prescribed allocation and execute.

This is expected to incur a high amount of gains as it trades indiscriminately.

Harvest Losses
--------------

This is a straight forward Harvesting algorithm.  Utilizing the Highest Cost tax lot release
methodology (HICO), we identify lots at a loss and sell them.  We subsequently invest the cash
raised, as well as any other cash available, to get us as close to our target allocation as we
can.  The only other caveat is that we must not buy back into the securities we just sold at a
loss in order to avoid a wash sale violation.

This is expected to incur many realized losses but should have difficulty tracking target
allocation in a market environment where securities are on the rise.

Avoiding Gains (Short and Long Term)
------------------------------------

This very similar to the Harvest Losses algorithm with the added restriction that we do not
harvest losses that take us further away from our target allocation and can only sell out of
securities at a loss up the target allocation for that security.

The expectations are very similar as for the Harvest Losses algorithm with only a modest
improvement in the ability to track target allocation.

Avoiding Short Term Gains
-------------------------

This takes the Avoiding Gains algorithm one step further by allowing the taking of long term gains
as they are taxed more favorably.  Furthermore, it is unlikely to get a more favorable taxation on
an embedded long term gain.  This leads to the sentiment that it's tolerable to take long term gains.

Implementation considerations:

- The tax lot release methodology employed for this algorithm is Minimum Short Term Gain (MSTG) and
requires a non-linear sort of the tax lots.  Within specific categories, we sort by descending cost
basis.  The category preference are laid out in the following order:
   * Short Term Losses
   * Long Term Losses
   * Long Term Gains
   * Short Term Gains
- For any of these algorithms, we must ensure that the reality of the tax lot release algorithm at
the custodial level, matches our assumptions or else our efforts are for naught.  If we execute a
trade with assumptions of MSTG but the custodian releases lots using HICO, then we will certainly
be realizing gains or losses we did not intend.
- This algorithm is special in that it requires we know the prices at the time of sale in order to
determine what the lot release order will be.

The expectation is that this enables the portfolio to easily track the changing target allocation.

Optimal Tax Lot Selection (NOT IMPLEMENTED)
-------------------------------------------

This algorithm would take into consideration the tax rates of the individual and combine that
with the depth of the gain or loss to determine which lot would be a better choice.  Consider
two lots with same number of shares.  One is a deep long term gain and the other is at a shallow
short term gain.  It may be more beneficial to take the short term gain depending on the price,
cost basis, and tax rates.

What's Missing?
===============

There are many dimensions left to explore, including:

- adequate specification of replacement securities
- modeling of client contributions and withdrawals
- use of other heuristics or algorithms to track a target allocation
- use of security volatility to determine the confidence that gain/loss status
  will persist until the execution of the trade
- other things I'm not thinking of.

This is a good start and is sufficient to demonstrate the power of using such algorithms.


Contents:

.. toctree::
   :maxdepth: 2

   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

