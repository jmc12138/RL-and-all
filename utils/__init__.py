
import os



try:
    from .others import *
except:
    from others import *

try:
    from .constant import *
except:
    from constant import *

try:
    from .train import *
except:
    from train import *



if __name__ == "__main__":
    play_with_keyboard('Breakout-v4')
