import numpy as np
import vectorbtpro as vbt

# Example data
close_price = np.array([99,99,95, 101, 95, 89, 105,100,96,115,101])
entries = np.array([False,True,False, False, False, False,False,False, False, False,False])
# We say that we can have at most 2 positions open at the same time
# so replicate the close_price and entries to have 2 columns
close_price = np.array([close_price, close_price]).T
entries = np.array([entries, entries]).T
entries[1,1] = False
entries[2,1] = True



# Allocate tsl_stop with the same shape as close_price
tsl_stop = np.full(close_price.shape, np.nan)
init_price = np.full(len(close_price), np.nan)
n_active_positions = np.full(len(close_price), np.nan)


# create empty list to store pos_info
pos_info_array = np.full(close_price.shape, np.nan)
n_pos = 2
pos_info_status_array = np.full(close_price.shape, np.nan)
asset_flow_array = np.full(close_price.shape, np.nan)
side_array = np.full(close_price.shape, np.nan)
order_price_array = np.full(close_price.shape, np.nan)

if False:
    # Define the custom adjust function
    @vbt.njit
    def adjust_func_nb(c, tsl_stop, n_active_positions, pos_info_array, pos_info_status_array, asset_flow_array):
        # Access the last position info
        n_active_positions[c.i] = vbt.pf_nb.get_n_active_positions_nb(c)
        asset_flow = vbt.pf_nb.get_allocation_nb(c) # vbt.pf_nb.get_col_order_count_nb(c)
        current_asset_flow = asset_flow[c.i, c.col]
        asset_flow_array[c.i, c.col] = current_asset_flow
        for j in range(n_pos):
            pos_info = c.last_pos_info[j]
            pos_info_array[c.i, j] = pos_info["entry_price"]
            # convert status to int
            if pos_info["status"] == vbt.pf_enums.TradeStatus.Open:
                pos_info_status_array[c.i, j] = 1 
            else:
                pos_info_status_array[c.i, j] = 0
        if n_active_positions[c.i] == 0: # if no positions are open
            tsl_stop[c.i, :] = 1 # do nothing
        else:
            for j in range(n_pos):
                # For each open trade, apply trailing stop logic
                if pos_info_status_array[c.i, j] == 1:
                    tsl_stop[c.i, j] = pos_info_array[c.i, j] * (1 - 0.1)  # Set tsl_stop at 10% below entry price
                else:
                    # If no new positions were opened, retain previous tsl_stop values
                    tsl_stop[c.i, j] = tsl_stop[c.i - 1, j]
            if pos_info_status_array[c.i, 0] == 1: # if new trade is opened
                tsl_stop[c.i, 0] = pos_info_array[c.i, 0] * (1 - 0.1)  # for the newest trade, we set the tsl_stop, remember, we do the -1, since we always have one active position here:
            elif pos_info_status_array[c.i, 1] == 1: # if new trade is opened
                tsl_stop[c.i, 1] = pos_info_array[c.i, 1] * (1 - 0.1)  # for the newest trade, we set the tsl_stop, remember, we do the -1, since we always have one active position here:
            elif pos_info_status_array[c.i, :].all() == 0: # then nothing happened, so we keep the tsl_stop the same
                tsl_stop[c.i,:] = tsl_stop[c.i-1,:]
         


@vbt.njit
def post_signal_func_nb(c, asset_flow_array, side_array, order_price_array, tsl_stop):
    '''
    This function is ONLY called after an order is executed.
    '''
    size = c.order_result.size
    asset_flow_array[c.i, c.col] = size
    side_array[c.i, c.col] = c.order_result.side
    order_price_array[c.i, c.col] = c.order_result.price
    if side_array[c.i, c.col] == 0:
        tsl_stop[c.i, c.col] = order_price_array[c.i, c.col] * (1 - 0.1)
    else:
        tsl_stop[c.i, c.col] = 1 

@vbt.njit
def adjust_func_nb(c, tsl_stop):
    '''
    This function is called to every price update.
    '''

    # we want to update the tsl_stop
    if c.last_position[c.col] == 1:
        # if current price is higher than the init_price, we want to update the tsl_stop
        if (c.last_pos_info["entry_price"][c.col] * 1.1) < c.close[c.i, c.col]: # TODO: 1.1 could be a parameter
            tsl_stop[c.i, c.col] = c.close[c.i, c.col] * (1 - 0.1)
        else:
            tsl_stop[c.i, c.col] = c.last_tsl_info["init_price"][c.col] * (1 - 0.1)
    else:
        tsl_stop[c.i, c.col] = np.nan

    # ensure that tsl_stop only increases as i increases
    if tsl_stop[c.i, c.col] < tsl_stop[c.i-1, c.col]:
        tsl_stop[c.i, c.col] = tsl_stop[c.i-1, c.col]

    tsl_info = c.last_tsl_info[c.col]
    sl_info = c.last_sl_info[c.col]
    if c.i > 0:
        # Set the trailing stop loss to a new value based on some logic
        tsl_info["stop"] = tsl_stop[c.i, c.col]  # set a new value
        sl_info.stop =  tsl_stop[c.i, c.col] / c.close[c.i, c.col] 
 
    pos_info = c.last_pos_info[c.col]
    if pos_info["status"] == vbt.pf_enums.TradeStatus.Open:
        if pos_info["return"] >= 0.1:
            sl_info = c.last_sl_info[c.col]
            vbt.pf_nb.set_sl_info_nb(
                sl_info, 
                init_idx=c.i, 
                # exit_size = 10,
                init_price=c.close[c.i, c.col]* (1-0.2),
                stop=tsl_stop[c.i, c.col],
                delta_format=vbt.pf_enums.DeltaFormat.Target
                )
        else:
            sl_info = c.last_sl_info[c.col]
            vbt.pf_nb.set_sl_info_nb(
                sl_info, 
                # exit_size = 10,
                init_idx=c.i, 
                init_price=c.close[c.i, c.col]* (1-0.1),
                stop=tsl_stop[c.i, c.col],
                delta_format=vbt.pf_enums.DeltaFormat.Target
                )

pf_cash = 100_000
# Create the portfolio
pf = vbt.PF.from_signals(
    close=close_price,
    size=1,
    init_cash=pf_cash,
    entries=entries,
    size_type="amount",
    tp_stop=0.2,
    # tsl_stop=tsl_stop, 
    sl_stop=np.inf,
    # tsl_stop=np.inf,
    stop_entry_price="fillprice",
    post_signal_func_nb=post_signal_func_nb,
    post_signal_args=(asset_flow_array, side_array, order_price_array, tsl_stop),
    adjust_func_nb=adjust_func_nb,
    adjust_args=(tsl_stop,),
    accumulate=True
)

print("pf.asset_flow: ")
print(pf.asset_flow)


print("asset_flow_array: ")
print(asset_flow_array)
print("side_array: ")
print(side_array)
print("order_price_array: ")
print(order_price_array)
print("tsl_stop: ")
print(tsl_stop)

# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(pf.get_cumulative_returns(group_by=True))
# plt.show()

try:
    fig = pf.iloc[:,0].get_trades().plot()
    i = 1
    fig = pf.iloc[:,i].get_trades().plot(plot_close=False, fig=fig).show()
except:
    print("plotting failed")