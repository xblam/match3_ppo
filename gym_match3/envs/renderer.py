import numpy as np
import cv2
import matplotlib
from gym_match3.envs.game import Board

matplotlib.use("TkAgg")

SQUARE_SIZE = 80
RENDER_SPEED = 1

class Renderer:
    def __init__(self, n_shapes):
        self.__n_shapes = n_shapes
        self.previousBoard = None
        self.images = []     
        for i in range(0, 17):
            self.images.append(cv2.imread(f"./gym_match3/envs/image/{i}.jpg", cv2.IMREAD_UNCHANGED))
        self.square_size = SQUARE_SIZE
        self.speed = RENDER_SPEED

    def render_board(self, board: Board, tiles=None):
        np_board = board.board
        img = np.zeros((np_board.shape[0]*self.square_size, np_board.shape[1]*self.square_size, 3), dtype=np.uint8)
    
        # Draw the initial board
        for i in range(np_board.shape[0]):
            for j in range(np_board.shape[1]):
                currentObject = int(np_board[i][j])
                small_image = cv2.resize(self.images[currentObject], (self.square_size, self.square_size))
                top_left = (j * self.square_size, i * self.square_size)
                bottom_right = (j * self.square_size + self.square_size, i * self.square_size + self.square_size)
                img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = small_image
                # Optionally, display text labels
                # cv2.putText(img, str(f'{i},{j}'), (j * self.square_size + self.square_size//2 - 40, i * self.square_size + self.square_size//2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)
    
        # Show the board with pieces before the swap
        cv2.imshow("board", img)
        cv2.waitKey(self.speed)
    
        if tiles is not None:
            x1, y1, x2, y2 = tiles['x1'], tiles['y1'], tiles['x2'], tiles['y2']
            first = int(np_board[x1][y1])
            second = int(np_board[x2][y2])
            imgFirst = cv2.resize(self.images[first], (self.square_size, self.square_size))
            imgSecond = cv2.resize(self.images[second], (self.square_size, self.square_size))

            # Show the animation of swapping pieces
            vertical = x1 != x2
            steps = self.square_size
            for i in range(0, steps + 1, 10):
                img[x1 * self.square_size:x1 * self.square_size + self.square_size,
                    y1 * self.square_size:y1 * self.square_size + self.square_size] = (255,255,255)
                img[x2 * self.square_size:x2 * self.square_size + self.square_size,
                    y2 * self.square_size:y2 * self.square_size + self.square_size] = (255,255,255)
                if vertical:
                    top_left1 = (y1 * self.square_size, x1 * self.square_size + i)
                    bottom_right1 = (y1 * self.square_size + self.square_size, x1 * self.square_size + self.square_size + i)
                    top_left2 = (y2 * self.square_size, x2 * self.square_size - i)
                    bottom_right2 = (y2 * self.square_size + self.square_size, x2 * self.square_size + self.square_size - i)
                else:
                    top_left1 = (y1 * self.square_size + i, x1 * self.square_size)
                    bottom_right1 = (y1 * self.square_size + self.square_size + i, x1 * self.square_size + self.square_size)
                    top_left2 = (y2 * self.square_size - i, x2 * self.square_size)
                    bottom_right2 = (y2 * self.square_size + self.square_size - i, x2 * self.square_size + self.square_size)

                img[top_left1[1]:bottom_right1[1], top_left1[0]:bottom_right1[0]] = imgFirst
                img[top_left2[1]:bottom_right2[1], top_left2[0]:bottom_right2[0]] = imgSecond
                cv2.imshow("board", img)
                cv2.waitKey(self.speed)  # Adjust speed as needed
        
            # Finally, place the pieces in their new positions
            # img[x1 * self.square_size:x1 * self.square_size + self.square_size, y1 * self.square_size:y1 * self.square_size + self.square_size] = imgSecond
            # img[x2 * self.square_size:x2 * self.square_size + self.square_size, y2 * self.square_size:y2 * self.square_size + self.square_size] = imgFirst
            # cv2.imshow("board", img)
            # cv2.waitKey(500)  # Show the final swapped positions for a short time