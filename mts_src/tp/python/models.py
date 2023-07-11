# simple moving average crossover model
# inputs: model_name	   - name of the model invocation as defined in the XML
#		  symbols          - 1xN list containing the traded instruments identified by MTS ticker
#         current_position - 1xN containing the current position in each traded instrument
#         timestamp		   - 1xN list of vectors with each vector containing the timestamp corresponding to a price bar
#         open_px          - 1xN lst of vectors with each vector containing open price bars in each traded instrument
#         high_px          - 1xN lst of vectors with each vector containing high price bars in each traded instrument
#         low_px           - 1xN lst of vectors with each vector containing low price bars in each traded instrument
#         close_px         - 1xN lst of vectors with each vector containing close price bars in each traded instrument
#         state_dict       - 1x1 dictionary of key-value pairs
#         data_dict        - 1X1 dictionary of key-value pairs populated from periodic database queries
# output: desired_position - 1xN list containing the target position in each traded instrument

state = {}
state['count'] = 0

def trend_model_sma(model_name, symbols, current_position, timestamp, open_px, high_px, low_px, close_px, state_dict, data_dict):

  import math

  desired_position = [];

  clip_size = 50

  numInst = len(symbols)

  for i in range(0,numInst):
    print("trend_model_sma")
    print(model_name)
    print(symbols[i])
    print(current_position[i])
    print(timestamp[i])
    print(close_px[i])
    print(state_dict)
    print(state)
    print(data_dict)
    print(len(state_dict))

    sma = sum(close_px[i])/len(close_px[i])
    signal = math.copysign(1,close_px[i][-1]-sma)
    desired_position.append(signal * clip_size)

    x_symbol = "test"
    x_timestamp = timestamp[i][0]
    x_px_close = close_px[i][0]
    print(f'{x_symbol}-{x_timestamp}')
    state_dict[f'{x_symbol}-{x_timestamp}'] = x_px_close

  state_dict['counter'] = state_dict['counter'] + 1
  state_dict['test'] = 42
  state['count'] = state['count'] + 1
  
  state_dict['traded_already'] = 0
  state_dict['instrument'] = 'WTI_1'
  state_dict['vol'] = 0.2344
  state_dict['trade_direction_coef'] = 1.0
  state_dict['trade_size'] = 200
  state_dict['strat_parameter_string'] = 'WTI_1/1.0/0.2344/200'
  print("done")
  
  return (desired_position, state_dict)


# simple breakout model
# inputs: model_name	   - name of the model invocation as defined in the XML
#		  symbols          - 1xN list containing the traded instruments identified by MTS ticker
#         current_position - 1xN containing the current position in each traded instrument
#         timestamp		   - 1xN list of vectors with each vector containing the timestamp corresponding to a price bar
#         open_px          - 1xN lst of vectors with each vector containing open price bars in each traded instrument
#         high_px          - 1xN lst of vectors with each vector containing high price bars in each traded instrument
#         low_px           - 1xN lst of vectors with each vector containing low price bars in each traded instrument
#         close_px         - 1xN lst of vectors with each vector containing close price bars in each traded instrument
#         state_dict       - 1x1 dictionary of key-value pairs
#         data_dict        - 1X1 dictionary of key-value pairs populated from periodic database queries
# output: desired_position - 1xN list containing the target position in each traded instrument

def trend_model_breakout(model_name, symbols, current_position, timestamp, open_px, high_px, low_px, close_px, state_dict, data_dict):

  desired_position = [];

  clip_size = 50

  numInst = len(symbols)

  for i in range(0,numInst):
    print("trend_model_breakout")
    print(model_name)
    print(symbols[i])
    print(current_position[i])
    print(timestamp[i])
    print(close_px[i])
    print(state_dict)
    print(state)
    print(data_dict)

    ll = min(close_px[i][0:-1])
    hh = max(close_px[i][0:-1])
    last_price = close_px[i][-1]

    target_pos = current_position[i]

    if target_pos <= 0 and last_price > hh:
      target_pos = clip_size

    if target_pos >= 0 and last_price < ll:
      target_pos = -clip_size

    desired_position.append(target_pos)

  state_dict['counter'] = state_dict['counter'] + 1

  return (desired_position, state_dict)

