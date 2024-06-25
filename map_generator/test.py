from map_generator import (
    Point,
    PartitionBigX,
    PartitionCenterPlus,
    PartitionTriangleVertical,
    PartitionSquare,
)

print("1", PartitionBigX().points)
print(PartitionCenterPlus().points)
print("3",PartitionTriangleVertical().points)
print(PartitionSquare(Point(0, 0), 3, 3).points)