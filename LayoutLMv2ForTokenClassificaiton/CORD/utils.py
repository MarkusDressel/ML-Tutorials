from typing import List
from numbers import Number

def normalize_bbox(bbox:List[Number], width:int, height:int) ->List[int]:
    """
    [summary]

    Parameters
    ----------
    bbox : List[Number]
        a list of number in the following order (x1,y1x3,y3)
    width : int
        original width of the image
    height : int
        original height of the image

    Returns
    -------
    List[int]
        the normilzed bounding box
    """
    return [
         int(1000 * (bbox[0] / width)),
         int(1000 * (bbox[1] / height)),
         int(1000 * (bbox[2] / width)),
         int(1000 * (bbox[3] / height)),
    ]
def unnormalize_box(bbox:List[Number], width:int, height:int):
    """
    rescales the bounding box from 1000 * 1000 to its original heights
    and width

   Parameters
    ----------
    bbox : List[Number]
        a list of number in the following order (x1,y1x3,y3)
    width : int
        original width of the image
    height : int
        original height of the image

    Returns
    -------
    List[int]
        the unnormilzed bounding box
    """
    return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
    ]