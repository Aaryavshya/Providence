## Updated version
import heapq
import cv2 as cv
import numpy as np
import math as m
import imutils
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

colors_in_img = set()

image = cv.imread(r'C:/Users/aravs/OneDrive/Desktop/grid.jpg')

grey_image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

blur = cv.GaussianBlur(grey_image, (5,5), 0)

threshold_image = cv.adaptiveThreshold(grey_image, 255, 1,1,11,2)
# cv.imshow("thres1", threshold_image)
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

mask = np.zeros((grey_image.shape),np.uint8)
cv.drawContours(mask,[best_cnt],0,255,-1)
cv.drawContours(mask,[best_cnt],0,0,2)

out = np.zeros_like(grey_image)
out[mask == 255] = grey_image[mask == 255]

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


# blur = cv.GaussianBlur(out, (5,5), 0)
# threshold_image = cv.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
# cv.imshow("thres", threshold_image)
# contours, _ = cv.findContours(threshold_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


# cropped_img = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
# image = cropped_img

# cv.imshow("image", cropped_img)

image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

h, s, v = cv.split(image_hsv)

# Increase saturation; you can adjust the multiplier as needed
saturated_s = s * 1.8   # Increase saturation
saturated_s = np.clip(saturated_s, 0, 255).astype(np.uint8)  # Ensure values are in 0-255 range

# Merge channels back and convert to BGR
saturated_image_hsv = cv.merge([h, saturated_s, v])
saturated_image_bgr = cv.cvtColor(saturated_image_hsv, cv.COLOR_HSV2BGR)
# cv.imshow("saturated", saturated_image_bgr)


# width = int(cropped_img.shape[0]/10)
width = int(image.shape[0]/10)

center = int(width/2)

# Color detection

color = []

for i in range(center,saturated_image_bgr.shape[0],width):
    for j in range(center,saturated_image_bgr.shape[0],width):
        color.append(saturated_image_bgr[i][j])

color = np.array(color)
color.reshape(10,10,3)


def color_distance(c1, c2):
    return np.sqrt(np.sum((c1 - c2)**2))

grid = color
# Flatten the grid to a list of colors for easier pairwise comparison
flat_grid = grid.reshape(-1, 3)
# Initialize the distance matrix
distance_matrix = np.zeros((100, 100))

# Calculate distances between each pair of squares
for i in range(100):
    for j in range(100):
        distance_matrix[i, j] = color_distance(flat_grid[i], flat_grid[j])

# Convert distances to a metric that can be correlated (if needed)
# This step depends on the specific kind of correlation you're looking for

inverse_distance_matrix = 1 / (1 + distance_matrix) # gives closeness 

# Calculate correlation matrix from the modified distance matrix
correlation_matrix = np.corrcoef(inverse_distance_matrix)

def classify_colors(rgb_matrix, n_clusters=5):
    colors = rgb_matrix.reshape(-1, 3)
    
    # Initialize the K-Means model with given number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    
    # Fit the model on the flattened color data
    kmeans.fit(colors)
    
    # Predict the cluster for each color in the matrix
    labels = kmeans.predict(colors)
    print(labels)
    
    # Reshape the labels back to the 10x10 matrix to get the classification of each color in the original structure
    labels_matrix = labels.reshape(10, 10)
    
    return np.rot90(labels_matrix, k=2, axes=(1, 0))

color_classes = classify_colors(color[::-1], n_clusters=6)
print(color_classes)
damage = []


# DAMAGE MAPPING

def map_to_damage(grid, mapping_dict):
    # Create an empty array with the same shape as the input grid to store the damage values
    damage_grid = np.zeros_like(grid, dtype=int)
    
    # Iterate over the grid and map each classifier number to its corresponding damage value
    for classifier, damage_value in mapping_dict.items():
        damage_grid[grid == classifier] = damage_value
    
    return damage_grid

mapping_dict = {
    0: 100, 
    1: 4,  # yellow
    2: 3,  # pink 
    3: 1,  # 
    4: 2, # Classifier 4 maps to damage value 50.0
    6: 100
}


damage = map_to_damage(color_classes, mapping_dict)

damage = np.array(damage)

print(np.array(damage).reshape(10, 10))

cv.waitKey(0)
cv.destroyAllWindows()


# PATHFINDING

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

#generating path string
def get_path_string(path):
    traverse = []
    for i in range(0,len(path),1):
        print(path[i])
        if path[i] == path[len(path)-1]:
            traverse.append(0)
            break
        if i+1 <= len(path):
            if (path[i][0] < path[i+1][0]) and ( path[i+1][0] - path[i-1][0] == 1 ) and ( path[i+1][1] - path[i-1][1] == 1 ):
                traverse.append(3)
                # traverse.append('s')
                # traverse.append('l')
            elif ((path[i][1] < path[i+1][1]) and ( path[i+1][0] - path[i-1][0] == 1 ) and ( path[i+1][1] - path[i-1][1] == 1 )) or ((path[i][1] > path[i+1][1]) and ( path[i-1][0] - path[i+1][0] == 1 ) and ( path[i-1][1] - path[i+1][1] == 1 )):
                traverse.append(2)
                # traverse.append('s')
                # traverse.append('r')
            elif (path[i][0] == path[i+1][0] and path[i][1] != path[i+1][1]) or (path[i][1] == path[i+1][1] and path[i][0] != path[i+1][0]):
                traverse.append(1)

    return traverse

def main(start, end):
    grid = np.flipud(np.array(damage).reshape(10,10))
    # grid[4, 4] = 0
    # grid[4, 5] = 0
    # grid[5, 4] = 0
    # grid[5, 5] = 0

    all_path_count = 0
    # start = (2, 9)
    # end = (4, 5)
    for least_health in range(100, 0, -1):
        path, final_health, paths_tried = lda_star(grid, start, end, least_health)
        all_path_count += paths_tried
        if path:
            path_string = get_path_string(path)
            print(len(path))
            print("Path:", path)
            print("Final Health:", final_health)
            print("Tried health:", least_health)
            print("Paths Tried:", all_path_count)
            draw_grid_with_path(grid, path, start, end)
            print("\n\n")
            print(path_string)
        
            

# start = [(0,0), (4, 4), (0, 2), (5, 4), (8, 1), (5, 5), (2, 9), (4, 5)]
# end = [(4,4), (0, 2), (5, 4), (8, 1), (5, 5), (2, 9), (4, 5), (9, 5)]

# path = []
# for i in range(len(start)):
#     print(f"start = {start}, end={end}")
#     path.append(main(start[i], end[i]))

# print(path)
main((9, 9), (3, 4))


