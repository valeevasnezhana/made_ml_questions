from typing import List, Optional

import numpy as np
from attr import define


@define
class LineImage:
    image_arr: np.ndarray
    line_list: List[np.ndarray]


@define
class PreparedImage:
    input_image: np.ndarray
    raw_sub_image_list: List[LineImage]
    line_image_list: List[LineImage]
    resized_line_image_list: List[LineImage]
    image_type: Optional[int] = None
    label: Optional[int] = None
