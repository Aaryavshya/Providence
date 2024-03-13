# import heapq
# import cv2 as cv
# import numpy as np
# import math as m
# import imutils

# # Cancel the red ones
# # yellow is the least damage ones
# # orange is the more damage ones
 
# image = cv.imread(r'C:/Users/aravs/OneDrive/Desktop/grid2.jpg')
# # cv.imshow("image",image)

# grey_image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
# # cv.imshow("grey",grey_image)

# blur = cv.GaussianBlur(grey_image, (5,5), 0)
# # cv.imshow("blur", blur)

# threshold_image = cv.adaptiveThreshold(grey_image, 255, 1,1,11,2)
# # cv.imshow("Threshold",threshold_image)

# contour, _ = cv.findContours(threshold_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# max_area = 0
# c = 0 

# for i in contour:
#     area = cv.contourArea(i)
#     if area > 1000:
#         if area > max_area:
#             max_area = area
#             best_cnt = i
#             image1 = cv.drawContours(image, contour, c, (255,0,0), 3)
#     c += 1

# # cv.imshow("Contour", image1)
# # print(contour)

# mask = np.zeros((grey_image.shape),np.uint8)
# cv.drawContours(mask,[best_cnt],0,255,-1)
# cv.drawContours(mask,[best_cnt],0,0,2)
# # cv.imshow("mask",mask)

# out = np.zeros_like(grey_image)
# out[mask == 255] = grey_image[mask == 255]
# # cv.imshow("Iteration",out)

# cnts = cv.findContours(threshold_image.copy(), cv.RETR_EXTERNAL,
# cv.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# c = max(cnts, key=cv.contourArea)
# extLeft = tuple(c[c[:, :, 0].argmin()][0])
# extRight = tuple(c[c[:, :, 0].argmax()][0])
# extTop = tuple(c[c[:, :, 1].argmin()][0])
# extBot = tuple(c[c[:, :, 1].argmax()][0])

# cv.drawContours(image, [c], -1, (0, 255, 255), 2)
# cv.circle(image, extLeft, 8, (0, 0, 255), -1)
# cv.circle(image, extRight, 8, (0, 255, 0), -1)
# cv.circle(image, extTop, 8, (255, 0, 0), -1)
# cv.circle(image, extBot, 8, (255, 255, 0), -1)
# # show the output image
# # cv.imshow("Image", image)

# blur = cv.GaussianBlur(out, (5,5), 0)
# # cv.imshow("blur1", blur)
# threshold_image = cv.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
# # cv.imshow("thresh1", threshold_image)

# contours, _ = cv.findContours(threshold_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# # c = 0
# # for i in contours:
# #         area = cv.contourArea(i)
# #         if area > 1000/2:
# #             cv.drawContours(image, contours, c, (0, 255, 0), 3)
# #         c+=1

# # cv.imshow("Final Image", image)
# # print(extTop, extBot, extLeft, extRight)
# cropped_img = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
# # cv.imshow("cropped_img", cropped_img[10:55,10:55])
# image = cropped_img

# image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

# # Split the HSV channels
# h, s, v = cv.split(image_hsv)

# # Increase saturation; you can adjust the multiplier as needed
# saturated_s = s * 1.5  # Increase saturation by 50%
# saturated_s = np.clip(saturated_s, 0, 255).astype(np.uint8)  # Ensure values are in 0-255 range

# # Merge channels back and convert to BGR
# saturated_image_hsv = cv.merge([h, saturated_s, v])
# saturated_image_bgr = cv.cvtColor(saturated_image_hsv, cv.COLOR_HSV2BGR)

# # Save or display the result
# # cv.imshow('Original Image', image)
# cv.imshow('Saturated Image', saturated_image_bgr)


# #cv.imshow("cropped_image",cropped_img)
# print(cropped_img.shape)
# width = int(cropped_img.shape[0]/10)
# print(width)
# # cv.imshow("one part",cropped_img[10:width,10:width])
# # print(cropped_img[width,width].shape)
# center = int(width/2)
# print(center)
# cv.imshow("Single",cropped_img[:center+center,:center+center])

# color = []

# for i in range(center,saturated_image_bgr.shape[0],width):
#     for j in range(center,saturated_image_bgr.shape[0],width):
#         color.append(saturated_image_bgr[i][j])

# color = np.array(color)
# color.reshape(10,10,3)
# print(color, len(color))

# def color_distance(c1, c2):
#     return np.sqrt(np.sum((c1 - c2)**2))

# grid = color
# # Flatten the grid to a list of colors for easier pairwise comparison
# flat_grid = grid.reshape(-1, 3)
# # print(flat_grid[0][8],flat_grid[0][9])
# # Initialize the distance matrix
# distance_matrix = np.zeros((100, 100))

# # Calculate distances between each pair of squares
# for i in range(100):
#     for j in range(100):
#         distance_matrix[i, j] = color_distance(flat_grid[i], flat_grid[j])

# # Convert distances to a metric that can be correlated (if needed)
# # This step depends on the specific kind of correlation you're looking for

# # Example: Inverse of distance (to simulate 'closeness')
# inverse_distance_matrix = 1 / (1 + distance_matrix)

# # Calculate correlation matrix from the modified distance matrix
# # Note: This is just an illustrative step; you'll need to define a clear basis for correlation
# correlation_matrix = np.corrcoef(inverse_distance_matrix)

# # print(correlation_matrix.shape)
# # print(correlation_matrix[0][8],correlation_matrix[0][9])

# # damage = [[]]

# # print(color[0])
# # print(len(color))

# bound = 65

# COLOR1 = (36, 39, 164)
# COLOR2 = (118, 64, 63)
# COLOR3 = (28, 148, 103)
# COLOR4 = (120, 66, 157)

# damage = []
# for i in range(0,len(color)):
#         if (0 <= color[i][0] < bound) and (0 <= color[i][1] < bound) and (0 <= color[i][2] < bound):
#             damage.append(100)
#         if ((color[i][0] > COLOR1[0] - bound and color[i][0] < COLOR1[0] + bound) and (color[i][1] > COLOR1[1] - bound and color[i][0] < COLOR1[1] + bound) and (color[i][2] > COLOR1[2] - bound and color[i][0] < COLOR1[2] + bound)):
#             damage.append(1)
#         if ((color[i][0] > COLOR2[0] - bound and color[i][0] < COLOR2[0] + bound ) and (color[i][1] > COLOR2[1] - bound and color[i][1] < COLOR2[1] + bound) and (color[i][2] > COLOR2[2] - bound and color[i][2] < COLOR2[2] + bound)):
#             damage.append(2)
#         if ((color[i][0] > COLOR3[0] - bound and color[i][0] < COLOR3[0] + bound) and (color[i][1] > COLOR3[1] - bound and color[i][1] < COLOR3[1] + bound ) and (color[i][2] > COLOR3[2] - bound and color[i][2] < COLOR3[2] + bound)):
#             damage.append(3)
#         if ((color[i][0] > COLOR4[0] - bound and color[i][0] < COLOR4[0] + bound) and (color[i][1] > COLOR4[1] - bound and color[i][1] < COLOR4[1] + bound ) and (color[i][2] > COLOR4[2] - bound and color[i][2] < COLOR4[2] + bound)):
#             damage.append(4)
#         # if (color[i][0] > 0 and color[i][0] < ) and (color[i][1] > 0 and color[i][1] < ) and (color[i][2] > 0 and color[i][2] < ):
#         #     damage.append(5)
#         if color[i][0] > 190 and color[i][1] > 190 and color[i][2] > 190:
#             damage.append(0)

# print(damage,len(damage)) 
# damage = np.array(damage)
# # damage.reshape(10, 10)
# for i in range(81):
#     if i%9 == 0:
#         print('\n')
#         print(damage[i], end=",")
#     else:
#         print(damage[i], end=",")

# # print(np.array(damage).reshape(10, 10))

# cv.waitKey(0)
# cv.destroyAllWindows()


# # # PATHFINDING

# class Node:
#     def __init__(self, position, parent=None):
#         self.position = position  # The node's location on the grid
#         self.parent = parent  # The parent node from which this node was expanded
#         self.g = 0  # Cost from the start node to this node
#         self.h = 0  # Estimated cost from this node to the goal
#         self.f = 0  # Total cost (g + h)
#         self.health = 100  # Node's health, starts at full
#         self.damage = 0  # Damage taken at this node

# # Estimates the cost from the current node to the goal, factoring in both distance and damage
# def heuristic(node, goal):
#     DiCF = 000000  # Distance Cost Factor
#     DCF = 100000  # Damage Cost Factor
#     hdi = m.sqrt((node.position[0] - goal.position[0]) ** 2 + (node.position[1] - goal.position[1]) ** 2)  # Euclidean distance
#     hd = node.damage  # Damage at the node
#     return DiCF * hdi + DCF * hd  # Combined heuristic

# # Generates neighbor nodes for a given node, considering valid, non-obstacle positions
# def get_neighbors(node, grid, start, end):
#     neighbors = []
#     for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  # Adjacent squares
#         node_position = (node.position[0] + new_position[0], node.position[1] + new_position[1])
#         # Ensure the position is within grid bounds and not an obstacle
#         if 0 <= node_position[0] < len(grid) and 0 <= node_position[1] < len(grid[0]) and grid[node_position[0]][node_position[1]] != 100:
#             new_node = Node(node_position, node)
#             # Damage is set to 0 for start and end nodes; for other nodes, it's determined by the grid value
#             new_node.damage = 0 if node_position in [start, end] else grid[node_position[0]][node_position[1]]
#             neighbors.append(new_node)
#     return neighbors

# # The LDA* algorithm, modified A* that also considers a node's remaining health
# def lda_star(grid, start, end, allowed_minimum_health):
#     start_node = Node(start)
#     end_node = Node(end)
#     open_list = []  # Nodes to be evaluated
#     closed_list = set()  # Nodes already evaluated
#     count = 0  # Counter for tie-breaking in the priority queue
#     paths_tried = 0  # Tracks the number of paths considered
#     tried_paths = []

#     heapq.heappush(open_list, (start_node.f, count, start_node))  # Initialize with the start node

#     while open_list:
#         _, _, current_node = heapq.heappop(open_list)
#         closed_list.add(current_node.position)  # Mark node as evaluated
#         paths_tried += 1  # Increment paths_tried for each node considered


#         # Check if the current node is the goal
#         if current_node.position == end_node.position:
#             path = []  # To store the path
#             final_health = current_node.health  # Capture the final health before altering the current_node reference
#             while current_node is not None:
#                 path.append(current_node.position)
#                 current_node = current_node.parent

#             return path[::-1], final_health, paths_tried  # Return the path, the captured final health, and paths tried

#         # Explore the neighbors of the current node
#         for neighbor in get_neighbors(current_node, grid, start, end):
#             if neighbor.position in closed_list or neighbor.health < allowed_minimum_health:
#                 continue  # Skip if already evaluated or health is below threshold

#             neighbor.g = current_node.g + 1  # Incremental cost from start to neighbor
#             neighbor.h = heuristic(neighbor, end_node)  # Estimated cost from neighbor to goal
#             neighbor.f = neighbor.g + neighbor.h  # Total cost
#             neighbor.health = current_node.health - neighbor.damage  # Update health after taking damage
#             # Only add this neighbor to the open list if it has a viable path (health-wise)
#             if neighbor.health >= allowed_minimum_health:
#                 existing_node = next((n for _, _, n in open_list if n.position == neighbor.position and n.health <= neighbor.health), None)
#                 if not existing_node or existing_node.f > neighbor.f:
#                     count += 1  # Increment tie-breaker
#                     heapq.heappush(open_list, (neighbor.f, count, neighbor))  # Add neighbor to open list

#     return None, None, paths_tried  # Return None if no path is found, along with the number of paths tried

# import matplotlib.pyplot as plt
# import numpy as np
# def draw_grid_with_path(grid, path, start, end):
#     grid_array=np.array(grid)
#     fig, ax=plt.subplots()
#     ax.set_aspect('equal')
#     ax.set_facecolor('white')
#     ax.axis('off')
#     for i in range(grid_array.shape[0]):
#         for j in range(grid_array.shape[1]):
#             cell_color='blue' if (i, j) in path else 'white'
#             ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, edgecolor='black', facecolor=cell_color))
#             ax.text(j, i, str(grid_array[i, j]), va='center', ha='center', color='black')
#     ax.plot(start[1], start[0], 'ro')
#     ax.plot(end[1], end[0], 'go')
#     plt.show()

# from random import randint
# def brute():
#     # grid = [
#     #     [4,2,4,1,1,4,4,1,3],
#     #     [2,4,2,4,1,100,3,4,100],
#     #     [1,4,100,4,2,4,3,2,1],
#     #     [1,4,100,4,2,4,3,2,1],

#     #     [4,2,4,1,1,2,5,3,2],

#     #     [100,4,2,5,100,3,2,1,4],

#     #     [1,1,4,3,2,1,100,3,5],

#     #     [3,2,3,1,1,3,3,100,1],

#     #     [2,4,100,1,3,4,100,2,1]
#     # ]
#     grid = np.array(damage).reshape(10, 10)
#     all_path_count = 0
#     start = (8, 0)

#     end = (0, 8)
#     for least_health in range(100, 0, -1):
#         path, final_health, paths_tried = lda_star(grid, start, end, least_health)
#         #print(f"Health: {least_health}, Paths Tried: {paths_tried}")
#         all_path_count += paths_tried
#         if path:
#             command = ""
#             for i in range(0, len(path) - 1):
#                 print(path[i], path[i+1])

#                 # if (path[i][0] == path[i+1][0]) and (path[i+1][1] - path[i][1] == 1):
#                 #     command += "ls"
#                 # if (path[i+1][0] - path[i][0] == 1) and (path[i][1] == path[i+1][1]):
#                 #     command += "rs"
#                 # if (path[i][0] - path[i+1][0] == 1) or (path[i][1] == path[i-1][1]):
#                 #     command += "s"
#                 if (path[i][0] != path[i+1][0]) or (path[i][1] != path[i+1][1]):

#                     if (path[i+1][0] - path[i][0] == 1) and (path[i][1] == path[i+1][1]):
#                         command += "lsr"
#                     if (path[i+1][0] == path[i][0]) and (path[i+1][1] - path[i][1] == 1):
#                         command += "rsl"
#                 # command += "s"
#                 print(command)
#             print("command", command)
#             print()
#             print("Path:", path)
#             print("Final Health:", final_health)
#             print("Tried health:", least_health)
#             print("Paths Tried:", all_path_count)
#             draw_grid_with_path(grid, path, start, end)
#             print("\n\n")
#             break
# brute()


## Updated version
import heapq
import cv2 as cv
import numpy as np
import math as m
import imutils
import webcolors
from sklearn.cluster import KMeans

colors_in_img = set()


def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour, spec=webcolors.CSS2)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name

# Cancel the red ones
# yellow is the least damage ones
# orange is the more damage ones

image = cv.imread(r'C:/Users/aravs/OneDrive/Desktop/grid2.jpg')
# cv.imshow("image",image)

grey_image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
# cv.imshow("grey",grey_image)

blur = cv.GaussianBlur(grey_image, (5,5), 0)
# cv.imshow("blur", blur)

threshold_image = cv.adaptiveThreshold(grey_image, 255, 1,1,11,2)
# cv.imshow("Threshold",threshold_image)

contour, _ = cv.findContours(threshold_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
max_area = 0
c = 0

for i in contour:
    area = cv.contourArea(i)
    if area > 1000:
        if area > max_area:
            max_area = area
            best_cnt = i
            image1 = cv.drawContours(image, contour, c, (255,0,0), 3)
    c += 1

# cv.imshow("Contour", image1)
# print(contour)

mask = np.zeros((grey_image.shape),np.uint8)
cv.drawContours(mask,[best_cnt],0,255,-1)
cv.drawContours(mask,[best_cnt],0,0,2)
# cv.imshow("mask",mask)

out = np.zeros_like(grey_image)
out[mask == 255] = grey_image[mask == 255]
# cv.imshow("Iteration",out)

cnts = cv.findContours(threshold_image.copy(), cv.RETR_EXTERNAL,
cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv.contourArea)
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])

cv.drawContours(image, [c], -1, (0, 255, 255), 2)
cv.circle(image, extLeft, 8, (0, 0, 255), -1)
cv.circle(image, extRight, 8, (0, 255, 0), -1)
cv.circle(image, extTop, 8, (255, 0, 0), -1)
cv.circle(image, extBot, 8, (255, 255, 0), -1)
# show the output image
# cv.imshow("Image", image)

blur = cv.GaussianBlur(out, (5,5), 0)
# cv.imshow("blur1", blur)
threshold_image = cv.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
# cv.imshow("thresh1", threshold_image)

contours, _ = cv.findContours(threshold_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# c = 0
# for i in contours:
#         area = cv.contourArea(i)
#         if area > 1000/2:
#             cv.drawContours(image, contours, c, (0, 255, 0), 3)
#         c+=1

# cv.imshow("Final Image", image)
# print(extTop, extBot, extLeft, extRight)
cropped_img = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
# cv.imshow("cropped_img", cropped_img[10:55,10:55])
image = cropped_img

image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

# Split the HSV channels
h, s, v = cv.split(image_hsv)

# Increase saturation; you can adjust the multiplier as needed
saturated_s = s * 1.8   # Increase saturation by 50%
saturated_s = np.clip(saturated_s, 0, 255).astype(np.uint8)  # Ensure values are in 0-255 range

# Merge channels back and convert to BGR
saturated_image_hsv = cv.merge([h, saturated_s, v])
saturated_image_bgr = cv.cvtColor(saturated_image_hsv, cv.COLOR_HSV2BGR)

# Save or display the result
# cv.imshow('Original Image', image)
# cv.imshow('Saturated Image', saturated_image_bgr)


cv.imshow("saturated", saturated_image_bgr)
# print(cropped_img.shape)
width = int(cropped_img.shape[0]/10)
# print(width)
# cv.imshow("one part",cropped_img[10:width,10:width])
# print(cropped_img[width,width].shape)
center = int(width/2)
# print(center)
# cv.imshow("Single",cropped_img[:center+center,:center+center])

color = []

for i in range(center,saturated_image_bgr.shape[0],width):
    for j in range(center,saturated_image_bgr.shape[0],width):
        color.append(saturated_image_bgr[i][j])

color = np.array(color)
color.reshape(10,10,3)
# print(color)

def color_distance(c1, c2):
    return np.sqrt(np.sum((c1 - c2)**2))

grid = color
# Flatten the grid to a list of colors for easier pairwise comparison
flat_grid = grid.reshape(-1, 3)
# print(flat_grid[0][8],flat_grid[0][9])
# Initialize the distance matrix
distance_matrix = np.zeros((100, 100))

# Calculate distances between each pair of squares
for i in range(100):
    for j in range(100):
        distance_matrix[i, j] = color_distance(flat_grid[i], flat_grid[j])

# Convert distances to a metric that can be correlated (if needed)
# This step depends on the specific kind of correlation you're looking for

# Example: Inverse of distance (to simulate 'closeness')
inverse_distance_matrix = 1 / (1 + distance_matrix)

# Calculate correlation matrix from the modified distance matrix
# Note: This is just an illustrative step; you'll need to define a clear basis for correlation
correlation_matrix = np.corrcoef(inverse_distance_matrix)

# print(correlation_matrix.shape)
# print(correlation_matrix[0][8],correlation_matrix[0][9])

# damage = [[]]

# print(color[0])
# print(len(color))

def classify_colors(rgb_matrix, n_clusters=5):
    # Reshape the 10x10x3 RGB matrix to a 100x3 matrix (100 colors, each represented by an RGB tuple)
    colors = rgb_matrix.reshape(-1, 3)
    
    # Initialize the K-Means model with 5 clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    
    # Fit the model on the flattened color data
    kmeans.fit(colors)
    
    # Predict the cluster for each color in the matrix
    labels = kmeans.predict(colors)
    
    # Reshape the labels back to the 10x10 matrix to get the classification of each color in the original structure
    labels_matrix = labels.reshape(10, 10)
    
    return np.rot90(labels_matrix, k=2, axes=(1, 0))

color_classes = classify_colors(color[::-1], n_clusters=6)
print(color_classes)
damage = []


def map_to_damage(grid, mapping_dict):
    # Create an empty array with the same shape as the input grid to store the damage values
    damage_grid = np.zeros_like(grid, dtype=int)
    
    # Iterate over the grid and map each classifier number to its corresponding damage value
    for classifier, damage_value in mapping_dict.items():
        damage_grid[grid == classifier] = damage_value
    
    return damage_grid

mapping_dict = {
    0: 1,  # Classifier 0 maps to damage value 10.0
    1: 1000,  # Classifier 1 maps to damage value 20.0
    2: 3,  # Classifier 2 maps to damage value 30.0
    3: 4,  # Classifier 3 maps to damage value 40.0
    4: 5,  # Classifier 4 maps to damage value 50.0
    5: 0
}


damage = map_to_damage(color_classes, mapping_dict)

print(damage)


print(colors_in_img, len(colors_in_img))
print(damage,len(damage))

damage = np.array(damage)


# for i in range(81):
#     if i%9 == 0:
#         print('\n')
#         print(damage[i], end=",")
#     else:
#         print(damage[i], end=",")

print(np.array(damage).reshape(10, 10))

cv.waitKey(0)
cv.destroyAllWindows()


# # PATHFINDING

class Node:
    def __init__(self, position, parent=None):
        self.position = position  # The node's location on the grid
        self.parent = parent  # The parent node from which this node was expanded
        self.g = 0  # Cost from the start node to this node
        self.h = 0  # Estimated cost from this node to the goal
        self.f = 0  # Total cost (g + h)
        self.health = 100  # Node's health, starts at full
        self.damage = 0  # Damage taken at this node

# Estimates the cost from the current node to the goal, factoring in both distance and damage
def heuristic(node, goal):
    DiCF = 000000  # Distance Cost Factor
    DCF = 10000000  # Damage Cost Factor
    hdi = m.sqrt((node.position[0] - goal.position[0]) ** 2 + (node.position[1] - goal.position[1]) ** 2)  # Euclidean distance
    hd = node.damage  # Damage at the node
    return DiCF * hdi + DCF * hd  # Combined heuristic

# Generates neighbor nodes for a given node, considering valid, non-obstacle positions
def get_neighbors(node, grid, start, end):
    neighbors = []
    for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  # Adjacent squares
        node_position = (node.position[0] + new_position[0], node.position[1] + new_position[1])
        # Ensure the position is within grid bounds and not an obstacle
        if 0 <= node_position[0] < len(grid) and 0 <= node_position[1] < len(grid[0]) and grid[node_position[0]][node_position[1]] != 100:
            new_node = Node(node_position, node)
            # Damage is set to 0 for start and end nodes; for other nodes, it's determined by the grid value
            new_node.damage = 0 if node_position in [start, end] else grid[node_position[0]][node_position[1]]
            neighbors.append(new_node)
    return neighbors

# The LDA* algorithm, modified A* that also considers a node's remaining health
def lda_star(grid, start, end, allowed_minimum_health):
    start_node = Node(start)
    end_node = Node(end)
    open_list = []  # Nodes to be evaluated
    closed_list = set()  # Nodes already evaluated
    count = 0  # Counter for tie-breaking in the priority queue
    paths_tried = 0  # Tracks the number of paths considered

    heapq.heappush(open_list, (start_node.f, count, start_node))  # Initialize with the start node

    while open_list:
        _, _, current_node = heapq.heappop(open_list)
        closed_list.add(current_node.position)  # Mark node as evaluated
        paths_tried += 1  # Increment paths_tried for each node considered


        # Check if the current node is the goal
        if current_node.position == end_node.position:
            path = []  # To store the path
            final_health = current_node.health  # Capture the final health before altering the current_node reference
            while current_node is not None:
                path.append(current_node.position)
                current_node = current_node.parent

            return path[::-1], final_health, paths_tried  # Return the path, the captured final health, and paths tried

        # Explore the neighbors of the current node
        for neighbor in get_neighbors(current_node, grid, start, end):
            if neighbor.position in closed_list or neighbor.health < allowed_minimum_health:
                continue  # Skip if already evaluated or health is below threshold

            neighbor.g = current_node.g + 1  # Incremental cost from start to neighbor
            neighbor.h = heuristic(neighbor, end_node)  # Estimated cost from neighbor to goal
            neighbor.f = neighbor.g + neighbor.h  # Total cost
            neighbor.health = current_node.health - neighbor.damage  # Update health after taking damage
            # Only add this neighbor to the open list if it has a viable path (health-wise)
            if neighbor.health >= allowed_minimum_health:
                existing_node = next((n for _, _, n in open_list if n.position == neighbor.position and n.health <= neighbor.health), None)
                if not existing_node or existing_node.f > neighbor.f:
                    count += 1  # Increment tie-breaker
                    heapq.heappush(open_list, (neighbor.f, count, neighbor))  # Add neighbor to open list

    return None, None, paths_tried  # Return None if no path is found, along with the number of paths tried

import matplotlib.pyplot as plt
import numpy as np
def draw_grid_with_path(grid, path, start, end):
    grid_array=np.array(grid)
    fig, ax=plt.subplots()
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    ax.axis('off')
    for i in range(grid_array.shape[0]):
        for j in range(grid_array.shape[1]):
            cell_color='blue' if (i, j) in path else 'white'
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, edgecolor='black', facecolor=cell_color))
            ax.text(j, i, str(grid_array[i, j]), va='center', ha='center', color='black')
    ax.plot(start[1], start[0], 'ro')
    ax.plot(end[1], end[0], 'go')
    plt.show()

#path string function
def get_path_string(path):
    traverse = []
    for i in range(0,len(path),1):
        print(path[i])
        if path[i] == path[len(path)-1]:
            traverse.append('h')
            break
        if i+1 <= len(path):
            if (path[i][0] < path[i+1][0]) and ( path[i+1][0] - path[i-1][0] == 1 ) and ( path[i+1][1] - path[i-1][1] == 1 ):
                traverse.append('l')
                # traverse.append('s')
                # traverse.append('l')
            elif (path[i][1] < path[i+1][1]) and ( path[i+1][0] - path[i-1][0] == 1 ) and ( path[i+1][1] - path[i-1][1] == 1 ):
                traverse.append('r')
                # traverse.append('s')
                # traverse.append('r')
            elif (path[i][0] == path[i+1][0] and path[i][1] != path[i+1][1]) or (path[i][1] == path[i+1][1] and path[i][0] != path[i+1][0]):
                traverse.append('s')

    print(traverse)

def brute():

    grid = np.flipud(np.array(damage).reshape(10,10))
    all_path_count = 0
    start = (0, 0)
    end = (9, 9)
    for least_health in range(100, 0, -1):
        path, final_health, paths_tried = lda_star(grid, start, end, least_health)
        #print(f"Health: {least_health}, Paths Tried: {paths_tried}")
        all_path_count += paths_tried
        if path:
            get_path_string(path)
            print()
            print("Path:", path)
            print("Final Health:", final_health)
            print("Tried health:", least_health)
            print("Paths Tried:", all_path_count)
            draw_grid_with_path(grid, path, start, end)
            print("\n\n")
            break
brute()


