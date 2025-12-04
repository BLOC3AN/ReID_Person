"""Bbox format conversion utilities"""

def bbox_to_coordinates(bbox):
    """
    Convert [x1,y1,x2,y2] to coordinate string format
    
    Args:
        bbox: [x1, y1, x2, y2] format
        
    Returns:
        str: "x1,y1;x2,y1;x2,y2;x1,y2"
    """
    x1, y1, x2, y2 = bbox
    return f"{x1},{y1};{x2},{y1};{x2},{y2};{x1},{y2}"


def bbox_to_points(bbox):
    """
    Convert [x1,y1,x2,y2] to list of 4 corner points
    
    Args:
        bbox: [x1, y1, x2, y2] format
        
    Returns:
        list: [(x1,y1), (x2,y1), (x2,y2), (x1,y2)]
    """
    x1, y1, x2, y2 = bbox
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
