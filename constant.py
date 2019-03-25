""" This file records constants """

BOX_SIZE = 6.5
K_T = 0.15
CUT_OFF = 3.0
FILE_SIZES = 10
ATOM_NUMBER = 50

DIRECTIONS = [[-BOX_SIZE, -BOX_SIZE, -BOX_SIZE],
              [0, -BOX_SIZE, -BOX_SIZE],
              [BOX_SIZE, -BOX_SIZE, -BOX_SIZE],
              [-BOX_SIZE, 0, -BOX_SIZE],
              [0, 0, -BOX_SIZE],
              [BOX_SIZE, 0, -BOX_SIZE],
              [-BOX_SIZE, BOX_SIZE, -BOX_SIZE],
              [0, BOX_SIZE, -BOX_SIZE],
              [BOX_SIZE, BOX_SIZE, -BOX_SIZE],
              [-BOX_SIZE, -BOX_SIZE, 0],
              [0, -BOX_SIZE, 0],
              [BOX_SIZE, -BOX_SIZE, 0],
              [-BOX_SIZE, 0, 0],
              [0, 0, 0],
              [BOX_SIZE, 0, 0],
              [-BOX_SIZE, BOX_SIZE, 0],
              [0, BOX_SIZE, 0],
              [BOX_SIZE, BOX_SIZE, 0],
              [-BOX_SIZE, -BOX_SIZE, BOX_SIZE],
              [0, -BOX_SIZE, BOX_SIZE],
              [BOX_SIZE, -BOX_SIZE, BOX_SIZE],
              [-BOX_SIZE, 0, BOX_SIZE],
              [0, 0, BOX_SIZE],
              [BOX_SIZE, 0, BOX_SIZE],
              [-BOX_SIZE, BOX_SIZE, BOX_SIZE],
              [0, BOX_SIZE, BOX_SIZE],
              [BOX_SIZE, BOX_SIZE, BOX_SIZE]]

RADIAL_SAMPLE_RUBRIC = {"Eta": [1.0],
                        "Rs": [0.5, 1.17, 1.83, 2.5]}

ANGULAR_SAMPLE_RUBRIC = {"Eta": [1.0, 2.0, 3.0],
                         "Rs": [0.5, 1.17, 1.83, 2.5],
                         "Zeta": [2.0, 4.0, 6.0],
                         "Thetas": [0.0, 1.57, 3.14, 4.71]}


