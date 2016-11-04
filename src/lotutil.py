"""Handy set of lot functions

-----
"""
import pandas as pd
import numpy as np
import simutil


def hico(lots, **kwargs):
    """Assuming one unique Id, return lots sorted
    according to Highest Cost for purposes of
    determining tax lot release.

    :param lots: DataFrame of lots
    :type lots: DataFrame

    :rtype: DataFrame
    """
    return lots.sort_values('Cost', ascending=False)


def lifo(lots, **kwargs):
    """Assuming one unique Id, return lots sorted
    according to Last In, First Out for purposes of
    determining tax lot release.

    :param lots: DataFrame of lots
    :type lots: DataFrame

    :rtype: DataFrame
    """
    return lots.sort_values('Date', ascending=False)


def mstg(lots, prices, date, **kwargs):
    """Minimum Short Term Gain is a lot release method that
    prioritizes Short Term losses > Long Term losses
    > Long Term gains > Short Term gains.

    I'll use pandas.Categorical to get the sorting correct

    :param lots: DataFrame of lots
    :type lots: DataFrame

    :param prices: time series of all prices
    :type prices: DataFrame

    :param date: The date to get prices as of
    :type date: Timestamp

    :rtype: DataFrame
    """
    l = add_term_gl_columns(lots, prices, date)
    drop_cols = ['Price', 'gl', 'term', 'term_gl']
    return l.sort_values(['term_gl', 'Cost'], ascending=False).drop(drop_cols, 1)


def add_term_gl_columns(lots, prices, date):
    date = pd.to_datetime(date)
    price = simutil.prices_asof(prices, date)
    year_ago = date - pd.Timedelta(days=366)

    l = lots.join(price.to_frame('Price'), on='Id')
    l['gl'] = to_cat(l.Cost.lt(l.Price), 'gain', 'loss')
    l['term'] = to_cat(l.Date.gt(year_ago), 'short', 'long')

    tg_cols = ['term', 'gl']
    cats = ['short_gain', 'long_gain', 'long_loss', 'short_loss']
    l['term_gl'] = pd.Categorical(l[tg_cols].apply('_'.join, 1), cats, True)
    return l


def pnl_df():
    """Return an empty profit and loss DataFrame

    :rtype: DataFrame
    """
    pnl_cols = ['Id', 'Quantity',
                'Cost', 'Sale', 'Bought',
                'Sold', 'Term', 'PNL', 'GL']
    pnl = pd.DataFrame([], columns=pnl_cols)
    return pnl


def sell_lots(lots, sale_quantity, sale_price, sale_date):
    """Given a set of lots, limited to a unique Id, an amount to sell,
    a price and a date, determine which lots to sell and return the
    remaining lots and the resulting gains and losses.

    :param lots: DataFrame of lots with a unique Id
    :type lots: DataFrame

    :param sale_quantity: Typically number of shares to sell
    :type sale_quantity: float

    :param sale_price: Price the sale transacted at
    :type sale_price: float

    :param sale_date: Date the sale transacted at
    :type sale_date: Timestamp

    :rtype: (DatFrame, DataFrame)
    """
    q = sale_quantity
    rows = lots.iterrows()

    pnl = pnl_df()

    while q < 0:
        idx, (id_, date, cost, purchase_quantity) = next(rows)
        remainder_quantity = max(0, purchase_quantity + q)
        lots.set_value(idx, 'Quantity', remainder_quantity)
        sale_quantity = purchase_quantity - remainder_quantity
        term = (sale_date - date).days >= 366
        gain_loss = sale_quantity * (sale_price - cost)
        gl = gain_loss > 0
        s = pd.Series([id_, sale_quantity,
                       cost, sale_price, date,
                       sale_date, term, gain_loss, gl],
                      pnl.columns)
        pnl = pnl.append(s, ignore_index=True)
        q += purchase_quantity
    pnl.Term = to_cat(pnl.Term, 'Long', 'Short')
    pnl.GL = to_cat(pnl.GL, 'Gain', 'Loss')
    return lots.query('Quantity > 0'), pnl


def build_lots(ledger, method=None, **kwargs):
    """Walk through the ledger and build lots as well as
    the consequential profit and loss and wash periods.

    :param ledger: DataFrame of account ledger
    :type ledger: DataFrame

    :param method: Function that sorts lots according to accounting method
    :type method: function

    :rtype: (DataFrame, DataFrame, DataFrame)
    """
    ledger = ledger[ledger.Transaction.isin(['buy', 'sell'])]

    if method is None:
        method = hico

    lot_cols = pd.Index(['Id', 'Date', 'Cost', 'Quantity'])
    lots = pd.DataFrame([], columns=lot_cols)

    pnl = pnl_df()

    wash = pd.DataFrame([], columns=['Id', 'Date', 'Exp', 'Restriction'])

    for (id_, date), (cost, quantity, transaction) in ledger.iterrows():
        if transaction == 'buy':
            s = pd.Series([id_, date, cost, quantity], lot_cols)
            lots = lots.append(s, ignore_index=True)
            exp = date + pd.Timedelta(days=30)
            wash = wash.append(
                pd.Series([id_, date, exp, 'sell'], wash.columns),
                ignore_index=True)
        else:
            lot_grps = lots.groupby(lots.Id.eq(id_))
            d = method(lot_grps.get_group(True), date=date, **kwargs)
            these_lots, these_pnl = sell_lots(d, quantity, cost, date)
            if these_pnl.PNL.lt(0).any():
                exp = date + pd.Timedelta(days=30)
                wash = wash.append(
                    pd.Series([id_, date, exp, 'buy'], wash.columns),
                    ignore_index=True)
            pnl = pnl.append(these_pnl, ignore_index=True)
            lots = pd.concat([
                    lot_grps.get_group(False),
                    these_lots
                ])
    return lots, pnl, collapse_wash_restrictions(wash)


def collapse_wash_restrictions(wash):
    """We can generate wash sale restrictions per transaction,
     but many of them may overlap.  This function collapses the
     periods for easier lookup later.

    :param wash: DataFrame with non-collapsed periods
    :type wash: DataFramesimutil.rst

    :rtype: DataFrame
    """

    def apply_wash_collapse(df):
        if len(df) == 1:
            return df
        else:
            s, e = 'Date', 'Exp'
            grps = df[s].gt(df[e].cummax().shift().bfill()).cumsum()
            funcs = {s: 'min', e: 'max', 'Id': 'first', 'Restriction': 'first'}
            return df.groupby(grps).agg(funcs).reindex_axis(df.columns, 1)

    g = wash.groupby(['Id', 'Restriction'])

    return g.apply(apply_wash_collapse).reset_index(drop=True)


def to_cat(bool_series, true_value, false_value):
    """Convert a boolean series to a series of categorical values

    :param bool_series: ``[True, False]`` series
    :type bool_series: Series

    :param true_value: Value when ``True``
    :type true_value: float or str

    :param false_value: Value when ``False``
    :type false_value: float or str

    :rtype: Series
    """
    return pd.Series(
        pd.Categorical(
            np.where(bool_series, true_value, false_value),
            [true_value, false_value], True
        ),
        bool_series.index
    )


def ledger_to_tlist(ledger):
    change = dict(Id='id_', Date='date',
                  Cost='cost', Quantity='quantity',
                  Transaction='transaction')
    tlist = ledger.reset_index().rename(columns=change)
    if 'transaction' not in tlist.columns:
        tlist['transaction'] = to_cat(tlist.quantity.lt(0), 'sell', 'buy')
    return tlist[['id_', 'date', 'cost', 'quantity', 'transaction']]

