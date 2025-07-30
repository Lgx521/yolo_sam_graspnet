#!/usr/bin/env python3

import sys
import os

# Add graspnet-baseline to Python path
graspnet_path = '/home/roar/graspnet/graspnet-baseline'
if graspnet_path not in sys.path:
    sys.path.insert(0, graspnet_path)

# Now import and run the actual service
from grasp_detection_service import main

if __name__ == '__main__':
    main()