import os
import sys
from datetime import datetime
from datetime import time

model_name = 'state_break'
loc = os.path.dirname(os.path.realpath(__file__))

# Logger - Disable in backtest
import logging
dtnow = datetime.now().strftime('%Y%m%d%H%M%S')
logging.basicConfig(filename=os.path.join(loc, f'{model_name}_{dtnow}.log'),
                    filemode='a',
                    level=logging.INFO,
                    format='%(asctime)s : %(levelname)s : %(message)s')

def state_break(
        model_name,
        symbols,
        current_position,
        timestamp,
        open_px,
        high_px,
        low_px,
        close_px,
        state_dict,
        data_dict):
    print(timestamp)
    print(state_dict)
    logger = logging.getLogger(__name__)
    try:
        for x_symbol_idx, x_symbol in enumerate(symbols):
            x_timestamp = timestamp[x_symbol_idx][0]
            x_px_close = close_px[x_symbol_idx][0]
            state_dict[f'{x_symbol}-{x_timestamp}'] = x_px_close
            
    except Exception as e:
        logger.error(e, exc_info=True)
        return (current_position, state_dict)
    
    return (current_position, state_dict)
