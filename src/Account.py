from pandas import DataFrame, Series, Index, Timestamp
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import re
import simutil, lotutil


def simulate(securities, tidx, yld, prc, account_name,
             sim_name, sim_method, verbose=False, seed=None,
             reb_freq='Q', allo_freq='2A', frac=1.,
             initial_value=100000000, marginal_rate=.25):

    """

    :param securities: List of security Id's
    :param tidx: Time series
    :param yld: Assumed dividends per period per security
    :param prc: Assumed initial price per security
    :param account_name: Name of account
    :param sim_name: Name of simulation
    :param sim_method: Algorithm
    :param verbose: Print simulation progress
    :param seed: random seed to pass to numpy
    :param reb_freq: Pandas string to use for resampling tidx
    :param allo_freq: How often allocation changes
    :param frac: What fraction of allocation weights to keep non-zero
    :param initial_value: Initial value of account
    :param marginal_rate: Marginal Tax Rate of account
    :rtype: (Account, DataFrame)
    """
    np.random.seed(seed)
    tidx = pd.to_datetime(tidx)
    reb_dates = tidx.to_series().resample('Q').last().index
    trets = simutil.randn_returns(securities, tidx)

    prices = simutil.implied_prices(trets, prc.add(yld), yld).dropna(1)
    allocations = simutil.random_allocation(prices, freq='2A', frac=1.)

    initial_date = tidx[0]
    initial_allocation = simutil.allocations_asof(allocations, initial_date)
    initial_shares = simutil.to_shares(initial_allocation,
                                       prices,
                                       initial_value,
                                       initial_date,
                                       2)

    acct = Account(account_name, marginal_rate)
    acct.cash_flow(initial_date, initial_value)

    initial_trade_list = acct.transaction_list(
        initial_allocation,
        prices,
        initial_date
    )

    acct.bulk_transact(initial_trade_list)

    for date in reb_dates:
        # allocation and price series for date
        allocation = simutil.allocations_asof(allocations, date)
        price = simutil.prices_asof(prices, date)
        # calculate number of days since last rebalance
        # and cash recieved from dividends
        lapsed_days = tidx.to_series().between(acct.last_transaction_date, date).sum()
        div_perd = yld.mul(acct.holdings()).sum()
        acct.dividend(date, lapsed_days * div_perd)

        harvest_flag = True
        if sim_method == 'avoid_short_gains':
            sells, do_not_buy = acct.avoid_short_gains_allocation(date, prices, allocation)
        elif sim_method == 'harvest_to_allocation':
            sells, do_not_buy = acct.harvest_to_allocation(date, prices, allocation)
        elif sim_method == 'harvest_losses':
            sells, do_not_buy = acct.harvest_losses(date, prices)
        else:
            acct.allocate(allocation, date, prices)
            harvest_flag = False
        if harvest_flag:
            acct.bulk_transact(sells)
            buys = acct.transaction_list_restrict_to_cash(allocation, prices, date, do_not_buy)
            acct.bulk_transact(buys)
        if verbose:
            print('\r', date.strftime('%Y-%m-%d'), end='')

    return acct, prices


class Account(object):
    """Account - class to handle the the ledger of transactions.

    Assume four types of transactions

    * ``'buy'`` - purchases
    * ``'sell'`` - sales
    * ``'flow'`` - cash withdrawal or contribution

    """
    def __init__(self, name, marginal_tax_rate, dividend_rate=.15, capital_gains_rate=.15):
        """Initialize Account

        :param name: required - expects one or more names separated by whitespace.
        :type name: str

        :param marginal_tax_rate: used to calculate impact of short term gain or loss
        :type marginal_tax_rate: float

        :param dividend_rate: used to calculate impact of dividends
        :type dividend_rate: float

        :param capital_gains_rate: used to calculate the impact of long term gain or loss
        :type capital_gains_rate: float

        :rtype: Account
        """
        self.name = name
        self.last_transaction_date = pd.NaT
        self.first_name = name.split()[0]
        self.last_name = name.split()[-1]
        self.mrg_rate = marginal_tax_rate
        self.div_rate = dividend_rate
        self.cap_rate = capital_gains_rate
        self.cash = 0
        self.column_values = ['Cost', 'Quantity', 'Transaction']
        self.column_index = Index(self.column_values, name='Fields')
        row_idx = pd.MultiIndex([[], []], [[], []], names=['Id', 'Date'])
        self.ledger = DataFrame([], row_idx, self.column_index)

    def transact(self, id_, date, cost, quantity, transaction):
        """Add a purchase or sale record to the ledger

        It's important to note that cost is used to determine capital gains or losses
        as well as determine how much the cash balance is affected.

        :param id_: The security/asset/holding identifier... Ticker
        :type id_: str

        :param date: Date of the transaction.
        :type date: Timestamp

        :param cost: Purchase cost
        :type cost: float

        :param quantity: Quantity purchased.  Typically thought of as shares
        :type quantity: float

        :param transaction: Type of transaction
        :type transaction: str

        :rtype: None
        """
        idx = (id_, pd.to_datetime(date))
        if quantity != 0:
            if transaction == 'flow':
                tran_val = cost * quantity
                if idx in self.ledger.index:
                    self.ledger.loc[idx, 'Quantity'] += quantity
                else:
                    self.ledger.loc[idx, :] = [cost, quantity, transaction]
                self.cash -= tran_val
            else:
                existing_tran_val = 0
                if idx in self.ledger.index:
                    existing_tran_val = self.ledger.get_value(idx, 'Cost') * self.ledger.get_value(idx, 'Quantity')
                new_tran_val = cost * quantity
                self.ledger.loc[(id_, pd.to_datetime(date)), :] = [cost, quantity, transaction]
                self.cash += existing_tran_val - new_tran_val
            self.last_transaction_date = self.ledger.index.get_level_values('Date').max()

    def bulk_transact(self, tlist):
        """Load transactions to ledger via a DataFrame

        column names match ``transact`` method arguments

        ``[id_, date, cost, quantity, transaction]``

        :param tlist: DataFrame of transactions
        """
        for i, row in tlist.dropna().query('quantity != 0').iterrows():
            self.transact(**row)

    def cash_flow(self, date, flow):
        """Add cash flow to the ledger and adjust cash

        :param date: Date of transaction
        :type date: Timestamp

        :param flow: Amount of cash flow
        :type flow: float

        """
        self.transact('_CSH', date, -1, flow, 'flow')

    def dividend(self, date, flow):
        """Add dividend payment to the ledger and adjust cash

        :param date: Date of transaction
        :type date: Timestamp

        :param flow: Amount of cash flow
        :type flow: float

        """
        self.transact('_CSH', date, -1, flow, 'flow')

    def buy(self, id_, date, cost, quantity):
        """Add purchase of ``id_`` to the ledger and adjust cash

        :param id_: The security/asset/holding identifier... Ticker
        :type id_: str

        :param date: Date of the transaction.
        :type date: Timestamp

        :param cost: Purchase cost
        :type cost: float

        :param quantity: Quantity purchased.  Typically thought of as shares
        :type quantity: float

        :rtype: None

        """
        assert cost >= 0, "Purchase cost must be greater than or equal to zero."
        assert quantity > 0, "Purchases must have quantity greater than zero."
        self.transact(id_, date, cost, quantity, 'buy')

    def sell(self, id_, date, cost, quantity):
        """Add sale of ``id_`` to the ledger and adjust cash

        :param id_: The security/asset/holding identifier... Ticker
        :type id_: str

        :param date: Date of the transaction.
        :type date: Timestamp

        :param cost: Purchase cost
        :type cost: float

        :param quantity: Quantity purchased.  Typically thought of as shares
        :type quantity: float

        :rtype: None

        """
        assert cost >= 0, "Purchase cost must be greater than or equal to zero."
        assert quantity < 0, "Purchases must have quantity less than zero."
        self.transact(id_, date, cost, quantity, 'sell')

    def holdings(self):
        """Return balances for all Id's and cash

        :rtype: Series
        """
        df = self.ledger.loc[self.ledger.Transaction.isin(['buy', 'sell']), :]
        h = df.groupby(level='Id').Quantity.sum()

        return h.append(pd.Series(dict(_CSH=self.cash)))

    def __getitem__(self, date):
        """Return Account with a ledger restricted to the given date and prior.

        :param date: Cut off date
        :type date: Timestamp

        :rtype: Account
        """
        if date >= self.last_transaction_date:
            return self
        else:
            date = pd.to_datetime(date)
            a = Account(self.name, self.mrg_rate, self.div_rate, self.cap_rate)
            change = dict(Id='id_', Date='date',
                          Cost='cost', Quantity='quantity',
                          Transaction='transaction')
            tlist = self.ledger[self.ledger.index.get_level_values('Date') <= date].reset_index()
            a.bulk_transact(tlist.rename(columns=change))
            return a

    def allocate(self, allocation, date, prices):
        """Perform necessary buys and sells to get close to prescribed weights
        defined in the allocation object.

        :param allocation: Series object with index containing id's and values containing weights
        :type allocation: Series

        :param date: Date the transactions are to take place
        :type date: Timestamp

        :param prices: Time series price reference
        :type prices: DataFrame

        :rtype: Account
        """
        date = pd.to_datetime(date)
        acct = self[date]
        acct.bulk_transact(acct.transaction_list(allocation, prices, date))
        return acct

    def __repr__(self):
        """String representation of Account object

        :rtype: str
        """
        fstr = 'First Name: {:>15s}\n' \
             + 'Last Name:  {:>15s}\n' \
             + 'Marginal:   {:>13.2f} %\n' \
             + '\n' \
             + 'Updated: {}\n' \
             + 'Cash: ${:,.2f}\n' \
             + 'Holdings: \n' \
             + '\n' \
             + '{}'
        holdings = re.sub(r'(^|\n)', r'\1    ', self.holdings().__repr__())
        return fstr.format(self.first_name,
                           self.last_name,
                           self.mrg_rate * 100,
                           self.last_transaction_date.strftime('%Y-%m-%d'),
                           self.cash,
                           holdings)

    def get_cash(self):
        """Return the cash value as a Series

        :rtype: Series
        """
        return pd.Series(self.cash, ['_CSH'], name='cash')

    def weights(self, date, prices):
        """Return the holdings (including cash) with respective weights

        :param date: the date to get holdings for
        :type date: Timestamp

        :param prices: DataFrame of prices
        :type prices: DataFrame

        :rtype: Series
        """
        date = pd.to_datetime(date)
        p = simutil.prices_asof(prices, date)
        h = self[date].holdings()
        w = p.mul(h)
        w.update(self.get_cash())
        return w.div(w.sum()).dropna()

    def market_value(self, date, prices):
        """Return the market value of holdings

        :param date: the date to get holdings for
        :type date: Timestamp

        :param prices: DataFrame of prices
        :type prices: DataFrame

        :rtype: float
        """
        date = pd.to_datetime(date)
        p = simutil.prices_asof(prices, date)
        h = self[date].holdings()
        v = p.mul(h)
        v.update(self.get_cash())
        return v.sum()

    def trades(self, share_allocation):
        """Take proposed shares and generate trade Series necessary to get there

        :param share_allocation: Series of proposed shares
        :type share_allocation: Series

        :rtype: Series
        """
        s = share_allocation
        return s.sub(self.holdings().drop('_CSH'), fill_value=0)

    def transaction_list(self, allocation, prices, date):
        """Generate list of transactions necessary to get to allocation weights

        :param allocation: Series of proposed weights
        :type allocation: Series

        :param prices: Time series of reference prices
        :type prices: DataFrame

        :param date: Date of reference
        :type date: Timestamp

        :rtype: DataFrame
        """
        date = pd.to_datetime(date)
        market_value = self.market_value(date, prices)
        share_allocation = simutil.to_shares(allocation, prices, market_value, date)
        price = simutil.prices_asof(prices, date)
        trades = self.trades(share_allocation)
        transaction = pd.Series(np.where(trades.lt(0), 'sell', 'buy'), trades.index)
        tlist = pd.concat([price, trades, transaction], join='inner',
                          axis=1, keys=['cost', 'quantity', 'transaction'])
        tlist.insert(0, 'date', date)
        return tlist.rename_axis('id_').reset_index().dropna().query('quantity != 0')

    def transaction_list_restrict_to_cash(self, allocation, prices, date, cant_buy=None):
        """Generate list of transactions using only cash that gets us
        closer to allocation weights

        :param allocation: Series of proposed weights
        :type allocation: Series

        :param prices: Time series of reference prices
        :type prices: DataFrame

        :param date: Date of reference
        :type date: Timestamp

        :param cant_buy: list of securities restricted by wash sale rules
        :type cant_buy: list

        :rtype: DataFrame
        """
        date = pd.to_datetime(date)
        if cant_buy is None:
            cant_buy = []

        market_value = self.market_value(date, prices)
        weights = self.weights(date, prices)
        price = simutil.prices_asof(prices, date)
        diff = allocation.sub(weights, fill_value=0)
        cash = -diff._CSH if '_CSH' in diff.index else 0
        buys = diff[(diff > 0) & ~diff.index.isin(cant_buy)]
        scaled_buys = buys.div(buys.sum()).mul(cash)
        share_buys = scaled_buys.mul(market_value).div(price) // 1
        transaction = lotutil.to_cat(share_buys.lt(0), 'sell', 'buy')
        tlist = pd.concat([price, share_buys, transaction], join='inner',
                          axis=1, keys=['cost', 'quantity', 'transaction'])
        tlist.insert(0, 'date', date)
        return tlist.rename_axis('id_').reset_index().dropna().query('quantity != 0')

    def historical_holdings(self):
        """Parse the ledger to produce a time series of historical shares.

        :rtype: DataFrame
        """
        cost = self.ledger.Cost.mul(self.ledger.Quantity)
        cash = cost.groupby(level='Date').sum().__neg__().cumsum().rename('_CSH')
        non_cash = self.ledger.Quantity.unstack(0, fill_value=0).drop('_CSH', 1).cumsum()
        return pd.concat([cash, non_cash], axis=1)

    def historical_market_values(self, prices):
        """Parse the ledger and prices to produce time series of historical market values

        :param prices: price reference
        :type prices: DataFrame

        :rtype: DataFrame
        """
        positions = self.historical_holdings()
        return positions.reindex_like(prices).ffill().mul(prices).dropna(1, 'all')

    def historical_weights(self, prices):
        """Parse the ledger and prices to produce time series of historical weights

        :param prices: price reference
        :type prices: DataFrame

        :rtype: DataFrame
        """
        market_values = self.historical_market_values(prices)
        return market_values.div(market_values.sum(1), 0)

    def historical_market_value(self, prices):
        """Parse the ledger and prices to produce time series of historical
         portfolio market value

        :param prices: price reference
        :type prices: DataFrame

        :rtype: DataFrame
        """
        return self.historical_market_values(prices).sum(1)

    def harvest_losses(self, date, prices, method=None):
        """Build tax lots, profit and loss, and wash sale restrictions
         with specified tax lot release method and return recommended trades.

         Determines trades to realize losses without violating wash sale restrictions.

        :param date: date of trade harvest execution
        :type date: Timestamp

        :param prices: reference prices
        :type prices: DataFrame

        :param method: function that sorts lots in release order
        :type method: function

        :rtype: DataFrame
        """
        date = pd.to_datetime(date)
        lots, pnl, wash = lotutil.build_lots(self[date].ledger, method=method)
        query_date = '(Date <= @date)'
        query_exp = '(Exp >= @date)'
        query_string = ' & '.join([query_date, query_exp])
        wash_ = wash.query(query_string)
        cant_sell = wash_[wash_.Restriction.eq('sell')].Id.unique()
        cant_buy = wash_[wash_.Restriction.eq('buy')].Id.unique()

        price = simutil.prices_asof(prices, date)
        lots_m = lots.merge(price.reset_index(name='Price'), on='Id')
        lots_l = lots_m.query('(Cost > Price) & (Id not in @cant_sell)')
        lots_h = lots_l.groupby('Id')[['Quantity']].sum().mul(-1)
        lots_h = lots_h.join(price.to_frame('Cost'))
        lots_h.insert(0, 'Date', date)
        lots_h = lots_h.reset_index().reindex_axis(lots.columns, 1)

        cant_buy = np.concatenate([cant_buy, lots_h.Id.unique()])

        return lotutil.ledger_to_tlist(lots_h), cant_buy

    def harvest_to_allocation(self, date, prices, allocation):
        """Build tax lots, profit and loss, and wash sale restrictions
        with Highest Cost tax lot release method and return recommended trades.
        In addition we restrict sells to what gets us closer to the
        allocation and not any further.

        Determines trades to realize losses without violating wash sale restrictions.

        Also, deliver list of securities we can not buy when we go to invest cash.

        :param date: date of trade harvest execution
        :type date: Timestamp

        :param prices: reference prices
        :type prices: DataFrame

        :param allocation: prescribed weights we are aiming at
        :type allocation: Series

        :param method: function that sorts lots in release order
        :type method: function

        :rtype: DataFrame
        """
        date = pd.to_datetime(date)
        lots, pnl, wash = lotutil.build_lots(self[date].ledger)
        query_date = '(Date <= @date)'
        query_exp = '(Exp >= @date)'
        query_string = ' & '.join([query_date, query_exp])
        wash_ = wash.query(query_string)
        cant_sell = wash_[wash_.Restriction.eq('sell')].Id.unique()
        cant_buy = wash_[wash_.Restriction.eq('buy')].Id.unique()

        price = simutil.prices_asof(prices, date)
        lots_m = lots.merge(price.reset_index(name='Price'), on='Id')
        lots_l = lots_m.query('(Cost > Price) & (Id not in @cant_sell)')
        lots_h = lots_l.groupby('Id')[['Quantity']].sum().mul(-1)
        lots_h = lots_h.join(price.to_frame('Cost'))
        lots_h.insert(0, 'Date', date)
        lots_h = lots_h.reset_index().reindex_axis(lots.columns, 1)

        h = lotutil.ledger_to_tlist(lots_h)
        t = self.transaction_list(allocation, prices, date).query('quantity < 0')
        h = h.merge(t[['id_', 'quantity']], on='id_', suffixes=['', '_'])
        h.loc[:, 'quantity'] = h.filter(like='quantity').max(1)
        h = h.drop('quantity_', 1)

        cant_buy = np.concatenate([cant_buy, h.id_.unique()])

        return h, cant_buy

    def avoid_short_gains_allocation(self, date, prices, allocation):
        """Build tax lots, profit and loss, and wash sale restrictions
        with Minimum Short Term Gain lot release method and return recommended trades
        that avoid short term gains.  This will allow long term gains.
        In addition we restrict sells to what gets us closer to the
        allocation and not any further.

        Determines trades to realize losses without violating wash sale restrictions.

        Also, deliver list of securities we can not buy when we go to invest cash.

        :param date: date of trade harvest execution
        :type date: Timestamp

        :param prices: reference prices
        :type prices: DataFrame

        :param allocation: prescribed weights we are aiming at
        :type allocation: Series

        :rtype: DataFrame
        """
        date = pd.to_datetime(date)
        lots, pnl, wash = lotutil.build_lots(self[date].ledger,
                                             method=lotutil.mstg,
                                             prices=prices)

        query_date = '(Date <= @date)'
        query_exp = '(Exp >= @date)'
        query_string = ' & '.join([query_date, query_exp])
        wash_ = wash.query(query_string)
        cant_sell = wash_[wash_.Restriction.eq('sell')].Id.unique()
        cant_buy = wash_[wash_.Restriction.eq('buy')].Id.unique()

        lot_cols = ['Id', 'Date', 'Cost', 'Quantity']
        lots = lotutil.add_term_gl_columns(lots, prices, date)
        lots_l = lots.query('(term_gl != "short_gain") & (Id not in @cant_sell)')
        lots_h = lots_l.groupby('Id')[['Quantity']].sum().mul(-1)
        lots_h = lots_h.join(simutil.prices_asof(prices, date).to_frame('Cost'))
        lots_h.insert(0, 'Date', date)
        lots_h = lots_h.reset_index().reindex_axis(lot_cols, 1)

        h = lotutil.ledger_to_tlist(lots_h)
        t = self.transaction_list(allocation, prices, date).query('quantity < 0')
        h = h.merge(t[['id_', 'quantity']], on='id_', suffixes=['', '_'])
        h.loc[:, 'quantity'] = h.filter(like='quantity').max(1)
        h = h.drop('quantity_', 1)

        cant_buy = np.concatenate([cant_buy, h.id_.unique()])

        return h, cant_buy

    def plot_weights(self, prices, title, figsize=(8, 4)):
        """Plot weights and market values

        :param prices: Time series of prices
        :type prices: DataFrame

        :param title: Title of charts
        :type title: str

        :param figsize: size of chart figure
        :type figsize: (int, int)
        """
        pkw = dict(stacked=True, colormap='jet')
        lkw = dict(loc='best', fancybox=True, framealpha=.5)
        skw = dict(left=0, bottom=0, right=1, top=.9, wspace=.05, hspace=0)

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(title, fontsize=18)

        mv = self.historical_market_values(prices).dropna(1, 'all')
        mv.plot.area(ax=axes[0], title='Market Value', **pkw)
        wt = mv.div(mv.sum(1), 0)
        wt.plot.area(ax=axes[1], title='Weight', **pkw)
        axes[1].yaxis.tick_right()

        for ax in axes:
            ax.legend(**lkw)

        plt.subplots_adjust(**skw)

    def pnl_table(self, method, **kwargs):
        """Generate 2x2 table of Long/Short Term Gains/Losses

        :param method: Tax lot release method
        :type method: function

        :rtype: DataFrame
        """
        lots, pnl, wash = lotutil.build_lots(self.ledger, method, **kwargs)
        pnl_tbl = pnl.groupby(['Term', 'GL']).PNL.sum().unstack(fill_value=0)
        return pnl_tbl.applymap('${:,.0f}'.format)
