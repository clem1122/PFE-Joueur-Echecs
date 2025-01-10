import chess

def create_compact_mapping():
    """
    Create a compact mapping of chess moves inspired by LC0's policy map.

    Returns:
        dict: A mapping from UCI strings to indices.
    """
    mapping = {}
    idx = 0

    # 1. Pawn moves (normal moves, captures, promotions)
    for file in 'abcdefgh':
        for rank in ['2', '7']:  # Starting ranks for pawns
            # Single forward move
            mapping[f"{file}{rank}{file}{int(rank)+1}"] = idx
            idx += 1
            # Double forward move
            mapping[f"{file}{rank}{file}{int(rank)+2}"] = idx
            idx += 1
        # Promotions
        for rank in ['7']:  # Promotion on 8th rank
            for promo in ['q', 'r', 'b', 'n']:
                mapping[f"{file}{rank}{file}8{promo}"] = idx
                idx += 1

    # 2. Knight moves
    knight_offsets = [
        (2, 1), (2, -1), (-2, 1), (-2, -1),
        (1, 2), (1, -2), (-1, 2), (-1, -2)
    ]
    for square in chess.SQUARES:
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        for dx, dy in knight_offsets:
            target_file = file + dx
            target_rank = rank + dy
            if 0 <= target_file < 8 and 0 <= target_rank < 8:  # Inside the board
                from_square = chess.square(file, rank)
                to_square = chess.square(target_file, target_rank)
                mapping[chess.SQUARE_NAMES[from_square] + chess.SQUARE_NAMES[to_square]] = idx
                idx += 1

    # 3. King moves
    king_offsets = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1)
    ]
    for square in chess.SQUARES:
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        for dx, dy in king_offsets:
            target_file = file + dx
            target_rank = rank + dy
            if 0 <= target_file < 8 and 0 <= target_rank < 8:  # Inside the board
                from_square = chess.square(file, rank)
                to_square = chess.square(target_file, target_rank)
                mapping[chess.SQUARE_NAMES[from_square] + chess.SQUARE_NAMES[to_square]] = idx
                idx += 1

    # 4. Castling
    mapping['e1g1'] = idx  # White kingside
    idx += 1
    mapping['e1c1'] = idx  # White queenside
    idx += 1
    mapping['e8g8'] = idx  # Black kingside
    idx += 1
    mapping['e8c8'] = idx  # Black queenside
    idx += 1

    # 5. Sliding pieces (rooks, bishops, queens)
    sliding_offsets = {
        'rook': [(1, 0), (-1, 0), (0, 1), (0, -1)],
        'bishop': [(1, 1), (1, -1), (-1, 1), (-1, -1)],
        'queen': [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ],
    }
    for piece, offsets in sliding_offsets.items():
        for square in chess.SQUARES:
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            for dx, dy in offsets:
                for step in range(1, 8):  # Extend the slide up to the board edge
                    target_file = file + step * dx
                    target_rank = rank + step * dy
                    if 0 <= target_file < 8 and 0 <= target_rank < 8:  # Inside the board
                        from_square = chess.square(file, rank)
                        to_square = chess.square(target_file, target_rank)
                        mapping[chess.SQUARE_NAMES[from_square] + chess.SQUARE_NAMES[to_square]] = idx
                        idx += 1
                    else:
                        break

    return mapping

# Generate the mapping
compact_mapping = create_compact_mapping()
#print(f"Total moves in the mapping: {len(compact_mapping)}")
mapping = compact_mapping
# from datasets import load_dataset
# dataset = load_dataset('angeluriot/chess_games')

absent_moves = set()
absent_moves = {'e7d8n', 'b2a1b', 'd2c1b', 'f2e1q', 'd7e8b', 'd2d1q', 'c7d8q', 'c7b8q', 'f7g8n', 'b2c1b', 'd2e1b', 'g2f1q', 'c2d1n', 'e7f8q',
'e2d1r', 'c7d8r', 'f2f1n', 'h2g1q', 'h2g1n', 'h2h1r', 'c2d1r', 'c2c1b', 'f2e1r', 'c2c1r', 'a7b8q', 'c2c1n', 'f7e8q', 'h2g1b', 'd7e8q', 'g2g1q',
'd2c1q', 'h7g8b', 'c7b8b', 'd2e1n', 'e2e1b', 'c7b8r', 'b7a8r', 'g2h1b', 'b2b1q', 'f2g1b', 'f7e8r', 'e2d1b', 'b7c8n', 'd7c8q', 'b7c8r', 'c2d1b',
'c2d1q', 'g7f8n', 'd2c1r', 'b7a8n', 'b2b1n', 'd2e1q', 'b2a1q', 'a2a1b', 'b2b1b', 'g7h8n', 'c7b8n', 'c2b1q', 'h2h1b', 'g2f1r', 'e2e1q', 'c7d8n',
'b2a1n', 'g2g1r', 'e2e1r', 'c2b1r', 'd2e1r', 'e2f1q', 'e2e1n', 'f7e8n', 'g7h8q', 'e2f1r', 'f2g1q', 'g2g1b', 'd2d1r', 'c2b1b', 'f2e1b', 'f2f1r',
'b2a1r', 'h7g8q', 'd7c8n', 'c2c1q', 'a7b8r', 'f2e1n', 'f7g8q', 'g2h1q', 'd7e8r', 'c2b1n', 'g2h1n', 'f2f1q', 'b7a8q', 'f7e8b', 'h2h1q', 'h7g8n',
'e7d8q', 'd2d1n', 'd2c1n', 'c7d8b', 'h2g1r', 'e7d8r', 'a2b1b', 'f7g8b', 'a7b8n', 'f7g8r', 'd2d1b', 'a2b1r', 'g2g1n', 'b2c1r', 'b7c8q', 'b7a8b',
'a2b1n', 'f2f1b', 'e7d8b', 'f2g1n', 'e2d1q', 'h2h1n', 'b2c1q', 'h7g8r', 'g7f8b', 'b7c8b', 'g7f8r', 'a2a1r', 'd7c8b', 'd7c8r', 'g7f8q', 'a2a1n',
'd7e8n', 'e7f8b', 'a2b1q', 'e2f1b', 'b2c1n', 'a2a1q', 'g7h8b', 'g2f1b', 'e7f8r', 'e2d1n', 'g2f1n', 'e7f8n', 'e2f1n', 'a7b8b', 'b2b1r', 'g2h1r', 'g7h8r', 'f2g1r'}
# for i, game in enumerate(dataset['train']):
#     moves_uci = game['moves_uci']
#     for move in moves_uci:
#         value = compact_mapping.get(move, -1)
#         if value == -1:
#             print(f"WARNING {move} not in mapping")
#             absent_moves.add(move)

#     print(f'game {i}')
#print(len(absent_moves))
#print("Absent moves:", absent_moves)


last_idx = max(compact_mapping.values()) if compact_mapping else 0
    
# Add each missing move to the mapping
for move in absent_moves:
    last_idx += 1  # Increment index for each missing move
    compact_mapping[move] = last_idx

#print(compact_mapping)

# mapping = {
#     'a2a3': 2455, 'a2a4': 2456, 'a7a8': 3395, 'a7a9': 3, 'a7a8q': 4, 'a7a8r': 5, 'a7a8b': 6, 'a7a8n': 7, 'b2b3': 2476, 'b2b4': 2477, 'b7b8': 3416,
#     'b7b9': 11, 'b7b8q': 12, 'b7b8r': 13, 'b7b8b': 14, 'b7b8n': 15, 'c2c3': 2499, 'c2c4': 2500, 'c7c8': 3439, 'c7c9': 19, 'c7c8q': 20, 'c7c8r': 21,
#     'c7c8b': 22, 'c7c8n': 23, 'd2d3': 2522, 'd2d4': 2523, 'd7d8': 3462, 'd7d9': 27, 'd7d8q': 28, 'd7d8r': 29, 'd7d8b': 30, 'd7d8n': 31, 'e2e3': 2545, 
#     'e2e4': 2546, 'e7e8': 3485, 'e7e9': 35, 'e7e8q': 36, 'e7e8r': 37, 'e7e8b': 38, 'e7e8n': 39, 'f2f3': 2568, 'f2f4': 2569, 'f7f8': 3508, 'f7f9': 43, 
#     'f7f8q': 44, 'f7f8r': 45, 'f7f8b': 46, 'f7f8n': 47, 'g2g3': 2591, 'g2g4': 2592, 'g7g8': 3531, 'g7g9': 51, 'g7g8q': 52, 'g7g8r': 53, 'g7g8b': 54, 
#     'g7g8n': 55, 'h2h3': 2614, 'h2h4': 2615, 'h7h8': 3554, 'h7h9': 59, 'h7h8q': 60, 'h7h8r': 61, 'h7h8b': 62, 'h7h8n': 63, 'a1c2': 64, 'a1b3': 65, 
#     'b1d2': 66, 'b1c3': 67, 'b1a3': 68, 'c1e2': 69, 'c1a2': 70, 'c1d3': 71, 'c1b3': 72, 'd1f2': 73, 'd1b2': 74, 'd1e3': 75, 'd1c3': 76, 'e1g2': 77, 
#     'e1c2': 78, 'e1f3': 79, 'e1d3': 80, 'f1h2': 81, 'f1d2': 82, 'f1g3': 83, 'f1e3': 84, 'g1e2': 85, 'g1h3': 86, 'g1f3': 87, 'h1f2': 88, 'h1g3': 89, 
#     'a2c3': 90, 'a2c1': 91, 'a2b4': 92, 'b2d3': 93, 'b2d1': 94, 'b2c4': 95, 'b2a4': 96, 'c2e3': 97, 'c2e1': 98, 'c2a3': 99, 'c2a1': 100, 'c2d4': 101, 
#     'c2b4': 102, 'd2f3': 103, 'd2f1': 104, 'd2b3': 105, 'd2b1': 106, 'd2e4': 107, 'd2c4': 108, 'e2g3': 109, 'e2g1': 110, 'e2c3': 111, 'e2c1': 112, 
#     'e2f4': 113, 'e2d4': 114, 'f2h3': 115, 'f2h1': 116, 'f2d3': 117, 'f2d1': 118, 'f2g4': 119, 'f2e4': 120, 'g2e3': 121, 'g2e1': 122, 'g2h4': 123, 
#     'g2f4': 124, 'h2f3': 125, 'h2f1': 126, 'h2g4': 127, 'a3c4': 128, 'a3c2': 129, 'a3b5': 130, 'a3b1': 131, 'b3d4': 132, 'b3d2': 133, 'b3c5': 134, 
#     'b3c1': 135, 'b3a5': 136, 'b3a1': 137, 'c3e4': 138, 'c3e2': 139, 'c3a4': 140, 'c3a2': 141, 'c3d5': 142, 'c3d1': 143, 'c3b5': 144, 'c3b1': 145, 
#     'd3f4': 146, 'd3f2': 147, 'd3b4': 148, 'd3b2': 149, 'd3e5': 150, 'd3e1': 151, 'd3c5': 152, 'd3c1': 153, 'e3g4': 154, 'e3g2': 155, 'e3c4': 156, 
#     'e3c2': 157, 'e3f5': 158, 'e3f1': 159, 'e3d5': 160, 'e3d1': 161, 'f3h4': 162, 'f3h2': 163, 'f3d4': 164, 'f3d2': 165, 'f3g5': 166, 'f3g1': 167, 
#     'f3e5': 168, 'f3e1': 169, 'g3e4': 170, 'g3e2': 171, 'g3h5': 172, 'g3h1': 173, 'g3f5': 174, 'g3f1': 175, 'h3f4': 176, 'h3f2': 177, 'h3g5': 178, 
#     'h3g1': 179, 'a4c5': 180, 'a4c3': 181, 'a4b6': 182, 'a4b2': 183, 'b4d5': 184, 'b4d3': 185, 'b4c6': 186, 'b4c2': 187, 'b4a6': 188, 'b4a2': 189, 
#     'c4e5': 190, 'c4e3': 191, 'c4a5': 192, 'c4a3': 193, 'c4d6': 194, 'c4d2': 195, 'c4b6': 196, 'c4b2': 197, 'd4f5': 198, 'd4f3': 199, 'd4b5': 200, 
#     'd4b3': 201, 'd4e6': 202, 'd4e2': 203, 'd4c6': 204, 'd4c2': 205, 'e4g5': 206, 'e4g3': 207, 'e4c5': 208, 'e4c3': 209, 'e4f6': 210, 'e4f2': 211, 
#     'e4d6': 212, 'e4d2': 213, 'f4h5': 214, 'f4h3': 215, 'f4d5': 216, 'f4d3': 217, 'f4g6': 218, 'f4g2': 219, 'f4e6': 220, 'f4e2': 221, 'g4e5': 222, 
#     'g4e3': 223, 'g4h6': 224, 'g4h2': 225, 'g4f6': 226, 'g4f2': 227, 'h4f5': 228, 'h4f3': 229, 'h4g6': 230, 'h4g2': 231, 'a5c6': 232, 'a5c4': 233, 
#     'a5b7': 234, 'a5b3': 235, 'b5d6': 236, 'b5d4': 237, 'b5c7': 238, 'b5c3': 239, 'b5a7': 240, 'b5a3': 241, 'c5e6': 242, 'c5e4': 243, 'c5a6': 244, 
#     'c5a4': 245, 'c5d7': 246, 'c5d3': 247, 'c5b7': 248, 'c5b3': 249, 'd5f6': 250, 'd5f4': 251, 'd5b6': 252, 'd5b4': 253, 'd5e7': 254, 'd5e3': 255, 
#     'd5c7': 256, 'd5c3': 257, 'e5g6': 258, 'e5g4': 259, 'e5c6': 260, 'e5c4': 261, 'e5f7': 262, 'e5f3': 263, 'e5d7': 264, 'e5d3': 265, 'f5h6': 266, 
#     'f5h4': 267, 'f5d6': 268, 'f5d4': 269, 'f5g7': 270, 'f5g3': 271, 'f5e7': 272, 'f5e3': 273, 'g5e6': 274, 'g5e4': 275, 'g5h7': 276, 'g5h3': 277, 
#     'g5f7': 278, 'g5f3': 279, 'h5f6': 280, 'h5f4': 281, 'h5g7': 282, 'h5g3': 283, 'a6c7': 284, 'a6c5': 285, 'a6b8': 286, 'a6b4': 287, 'b6d7': 288, 
#     'b6d5': 289, 'b6c8': 290, 'b6c4': 291, 'b6a8': 292, 'b6a4': 293, 'c6e7': 294, 'c6e5': 295, 'c6a7': 296, 'c6a5': 297, 'c6d8': 298, 'c6d4': 299, 
#     'c6b8': 300, 'c6b4': 301, 'd6f7': 302, 'd6f5': 303, 'd6b7': 304, 'd6b5': 305, 'd6e8': 306, 'd6e4': 307, 'd6c8': 308, 'd6c4': 309, 'e6g7': 310, 
#     'e6g5': 311, 'e6c7': 312, 'e6c5': 313, 'e6f8': 314, 'e6f4': 315, 'e6d8': 316, 'e6d4': 317, 'f6h7': 318, 'f6h5': 319, 'f6d7': 320, 'f6d5': 321, 
#     'f6g8': 322, 'f6g4': 323, 'f6e8': 324, 'f6e4': 325, 'g6e7': 326, 'g6e5': 327, 'g6h8': 328, 'g6h4': 329, 'g6f8': 330, 'g6f4': 331, 'h6f7': 332, 
#     'h6f5': 333, 'h6g8': 334, 'h6g4': 335, 'a7c8': 336, 'a7c6': 337, 'a7b5': 338, 'b7d8': 339, 'b7d6': 340, 'b7c5': 341, 'b7a5': 342, 'c7e8': 343, 
#     'c7e6': 344, 'c7a8': 345, 'c7a6': 346, 'c7d5': 347, 'c7b5': 348, 'd7f8': 349, 'd7f6': 350, 'd7b8': 351, 'd7b6': 352, 'd7e5': 353, 'd7c5': 354, 
#     'e7g8': 355, 'e7g6': 356, 'e7c8': 357, 'e7c6': 358, 'e7f5': 359, 'e7d5': 360, 'f7h8': 361, 'f7h6': 362, 'f7d8': 363, 'f7d6': 364, 'f7g5': 365, 
#     'f7e5': 366, 'g7e8': 367, 'g7e6': 368, 'g7h5': 369, 'g7f5': 370, 'h7f8': 371, 'h7f6': 372, 'h7g5': 373, 'a8c7': 374, 'a8b6': 375, 'b8d7': 376, 
#     'b8c6': 377, 'b8a6': 378, 'c8e7': 379, 'c8a7': 380, 'c8d6': 381, 'c8b6': 382, 'd8f7': 383, 'd8b7': 384, 'd8e6': 385, 'd8c6': 386, 'e8g7': 387, 
#     'e8c7': 388, 'e8f6': 389, 'e8d6': 390, 'f8h7': 391, 'f8d7': 392, 'f8g6': 393, 'f8e6': 394, 'g8e7': 395, 'g8h6': 396, 'g8f6': 397, 'h8f7': 398, 
#     'h8g6': 399, 'a1b1': 2280, 'a1a2': 2287, 'a1b2': 2294, 'b1c1': 2301, 'b1a1': 2307, 'b1b2': 2308, 'b1c2': 2315, 'b1a2': 2321, 'c1d1': 2322, 
#     'c1b1': 2327, 'c1c2': 2329, 'c1d2': 2336, 'c1b2': 2341, 'd1e1': 2343, 'd1c1': 2347, 'd1d2': 2350, 'd1e2': 2357, 'd1c2': 2361, 'e1f1': 2364, 
#     'e1d1': 2367, 'e1e2': 2371, 'e1f2': 2378, 'e1d2': 2381, 'f1g1': 2385, 'f1e1': 2387, 'f1f2': 2392, 'f1g2': 2399, 'f1e2': 2401, 'g1h1': 2406,
#      'g1f1': 2407, 'g1g2': 2413, 'g1h2': 2420, 'g1f2': 2421, 'h1g1': 2427, 'h1h2': 2434, 'h1g2': 2441, 'a2b2': 2448, 'a2a1': 2461, 'a2b3': 2462, 
#      'a2b1': 2468, 'b2c2': 2469, 'b2a2': 2475, 'b2b1': 2482, 'b2c3': 2483, 'b2c1': 2489, 'b2a3': 2490, 'b2a1': 2491, 'c2d2': 2492, 'c2b2': 2497, 
#      'c2c1': 2505, 'c2d3': 2506, 'c2d1': 2511, 'c2b3': 2512, 'c2b1': 2514, 'd2e2': 2515, 'd2c2': 2519, 'd2d1': 2528, 'd2e3': 2529, 'd2e1': 2533, 
#      'd2c3': 2534, 'd2c1': 2537, 'e2f2': 2538, 'e2d2': 2541, 'e2e1': 2551, 'e2f3': 2552, 'e2f1': 2555, 'e2d3': 2556, 'e2d1': 2560, 'f2g2': 2561, 
#      'f2e2': 2563, 'f2f1': 2574, 'f2g3': 2575, 'f2g1': 2577, 'f2e3': 2578, 'f2e1': 2583, 'g2h2': 2584, 'g2f2': 2585, 'g2g1': 2597, 'g2h3': 2598, 
#      'g2h1': 2599, 'g2f3': 2600, 'g2f1': 2606, 'h2g2': 2607, 'h2h1': 2620, 'h2g3': 2621, 'h2g1': 2627, 'a3b3': 2628, 'a3a4': 2635, 'a3a2': 2640, 
#      'a3b4': 2642, 'a3b2': 2647, 'b3c3': 2649, 'b3a3': 2655, 'b3b4': 2656, 'b3b2': 2661, 'b3c4': 2663, 'b3c2': 2668, 'b3a4': 2670, 'b3a2': 2671, 
#      'c3d3': 2672, 'c3b3': 2677, 'c3c4': 2679, 'c3c2': 2684, 'c3d4': 2686, 'c3d2': 2691, 'c3b4': 2693, 'c3b2': 2695, 'd3e3': 2697, 'd3c3': 2701, 
#      'd3d4': 2704, 'd3d2': 2709, 'd3e4': 2711, 'd3e2': 2715, 'd3c4': 2717, 'd3c2': 2720, 'e3f3': 2722, 'e3d3': 2725, 'e3e4': 2729, 'e3e2': 2734, 
#      'e3f4': 2736, 'e3f2': 2739, 'e3d4': 2741, 'e3d2': 2745, 'f3g3': 2747, 'f3e3': 2749, 'f3f4': 2754, 'f3f2': 2759, 'f3g4': 2761, 'f3g2': 2763, 
#      'f3e4': 2765, 'f3e2': 2770, 'g3h3': 2772, 'g3f3': 2773, 'g3g4': 2779, 'g3g2': 2784, 'g3h4': 2786, 'g3h2': 2787, 'g3f4': 2788, 'g3f2': 2793, 
#      'h3g3': 2795, 'h3h4': 2802, 'h3h2': 2807, 'h3g4': 2809, 'h3g2': 2814, 'a4b4': 2816, 'a4a5': 2823, 'a4a3': 2827, 'a4b5': 2830, 'a4b3': 2834, 
#      'b4c4': 2837, 'b4a4': 2843, 'b4b5': 2844, 'b4b3': 2848, 'b4c5': 2851, 'b4c3': 2855, 'b4a5': 2858, 'b4a3': 2859, 'c4d4': 2860, 'c4b4': 2865, 
#      'c4c5': 2867, 'c4c3': 2871, 'c4d5': 2874, 'c4d3': 2878, 'c4b5': 2881, 'c4b3': 2883, 'd4e4': 2885, 'd4c4': 2889, 'd4d5': 2892, 'd4d3': 2896, 
#      'd4e5': 2899, 'd4e3': 2903, 'd4c5': 2906, 'd4c3': 2909, 'e4f4': 2912, 'e4d4': 2915, 'e4e5': 2919, 'e4e3': 2923, 'e4f5': 2926, 'e4f3': 2929, 
#      'e4d5': 2932, 'e4d3': 2936, 'f4g4': 2939, 'f4e4': 2941, 'f4f5': 2946, 'f4f3': 2950, 'f4g5': 2953, 'f4g3': 2955, 'f4e5': 2957, 'f4e3': 2961, 
#      'g4h4': 2964, 'g4f4': 2965, 'g4g5': 2971, 'g4g3': 2975, 'g4h5': 2978, 'g4h3': 2979, 'g4f5': 2980, 'g4f3': 2984, 'h4g4': 2987, 'h4h5': 2994, 
#      'h4h3': 2998, 'h4g5': 3001, 'h4g3': 3005, 'a5b5': 3008, 'a5a6': 3015, 'a5a4': 3018, 'a5b6': 3022, 'a5b4': 3025, 'b5c5': 3029, 'b5a5': 3035, 
#      'b5b6': 3036, 'b5b4': 3039, 'b5c6': 3043, 'b5c4': 3046, 'b5a6': 3050, 'b5a4': 3051, 'c5d5': 3052, 'c5b5': 3057, 'c5c6': 3059, 'c5c4': 3062, 
#      'c5d6': 3066, 'c5d4': 3069, 'c5b6': 3073, 'c5b4': 3075, 'd5e5': 3077, 'd5c5': 3081, 'd5d6': 3084, 'd5d4': 3087, 'd5e6': 3091, 'd5e4': 3094, 
#     'd5c6': 3098, 'd5c4': 3101, 'e5f5': 3104, 'e5d5': 3107, 'e5e6': 3111, 'e5e4': 3114, 'e5f6': 3118, 'e5f4': 3121, 'e5d6': 3124, 'e5d4': 3127, 
#     'f5g5': 3131, 'f5e5': 3133, 'f5f6': 3138, 'f5f4': 3141, 'f5g6': 3145, 'f5g4': 3147, 'f5e6': 3149, 'f5e4': 3152, 'g5h5': 3156, 'g5f5': 3157, 
#     'g5g6': 3163, 'g5g4': 3166, 'g5h6': 3170, 'g5h4': 3171, 'g5f6': 3172, 'g5f4': 3175, 'h5g5': 3179, 'h5h6': 3186, 'h5h4': 3189, 'h5g6': 3193, 
#     'h5g4': 3196, 'a6b6': 3200, 'a6a7': 3207, 'a6a5': 3209, 'a6b7': 3214, 'a6b5': 3216, 'b6c6': 3221, 'b6a6': 3227, 'b6b7': 3228, 'b6b5': 3230, 
#     'b6c7': 3235, 'b6c5': 3237, 'b6a7': 3242, 'b6a5': 3243, 'c6d6': 3244, 'c6b6': 3249, 'c6c7': 3251, 'c6c5': 3253, 'c6d7': 3258, 'c6d5': 3260, 
#     'c6b7': 3265, 'c6b5': 3267, 'd6e6': 3269, 'd6c6': 3273, 'd6d7': 3276, 'd6d5': 3278, 'd6e7': 3283, 'd6e5': 3285, 'd6c7': 3289, 'd6c5': 3291, 
#     'e6f6': 3294, 'e6d6': 3297, 'e6e7': 3301, 'e6e5': 3303, 'e6f7': 3308, 'e6f5': 3310, 'e6d7': 3313, 'e6d5': 3315, 'f6g6': 3319, 'f6e6': 3321, 
#     'f6f7': 3326, 'f6f5': 3328, 'f6g7': 3333, 'f6g5': 3335, 'f6e7': 3337, 'f6e5': 3339, 'g6h6': 3344, 'g6f6': 3345, 'g6g7': 3351, 'g6g5': 3353, 
#     'g6h7': 3358, 'g6h5': 3359, 'g6f7': 3360, 'g6f5': 3362, 'h6g6': 3367, 'h6h7': 3374, 'h6h5': 3376, 'h6g7': 3381, 'h6g5': 3383, 'a7b7': 3388, 
#     'a7a6': 3396, 'a7b8': 3402, 'a7b6': 3403, 'b7c7': 3409, 'b7a7': 3415, 'b7b6': 3417, 'b7c8': 3423, 'b7c6': 3424, 'b7a8': 3430, 'b7a6': 3431, 
#     'c7d7': 3432, 'c7b7': 3437, 'c7c6': 3440, 'c7d8': 3446, 'c7d6': 3447, 'c7b8': 3452, 'c7b6': 3453, 'd7e7': 3455, 'd7c7': 3459, 'd7d6': 3463,
#     'd7e8': 3469, 'd7e6': 3470, 'd7c8': 3474, 'd7c6': 3475, 'e7f7': 3478, 'e7d7': 3481, 'e7e6': 3486, 'e7f8': 3492, 'e7f6': 3493, 'e7d8': 3496, 
#     'e7d6': 3497, 'f7g7': 3501, 'f7e7': 3503, 'f7f6': 3509, 'f7g8': 3515, 'f7g6': 3516, 'f7e8': 3518, 'f7e6': 3519, 'g7h7': 3524, 'g7f7': 3525, 
#     'g7g6': 3532, 'g7h8': 3538, 'g7h6': 3539, 'g7f8': 3540, 'g7f6': 3541, 'h7g7': 3547, 'h7h6': 3555, 'h7g8': 3561, 'h7g6': 3562, 'a8b8': 3568, 
#     'a8a7': 3575, 'a8b7': 3582, 'b8c8': 3589, 'b8a8': 3595, 'b8b7': 3596, 'b8c7': 3603, 'b8a7': 3609, 'c8d8': 3610, 'c8b8': 3615, 'c8c7': 3617, 
#     'c8d7': 3624, 'c8b7': 3629, 'd8e8': 3631, 'd8c8': 3635, 'd8d7': 3638, 'd8e7': 3645, 'd8c7': 3649, 'e8f8': 3652, 'e8d8': 3655, 'e8e7': 3659, 
#     'e8f7': 3666, 'e8d7': 3669, 'f8g8': 3673, 'f8e8': 3675, 'f8f7': 3680, 'f8g7': 3687, 'f8e7': 3689, 'g8h8': 3694, 'g8f8': 3695, 'g8g7': 3701, 
#     'g8h7': 3708, 'g8f7': 3709, 'h8g8': 3715, 'h8h7': 3722, 'h8g7': 3729, 'e1g1': 2365, 'e1c1': 2368, 'e8g8': 3653, 'e8c8': 3656, 'a1c1': 2281, 
#     'a1d1': 2282, 'a1e1': 2283, 'a1f1': 2284, 'a1g1': 2285, 'a1h1': 2286, 'a1a3': 2288, 'a1a4': 2289, 'a1a5': 2290, 'a1a6': 2291, 'a1a7': 2292, 
#     'a1a8': 2293, 'b1d1': 2302, 'b1e1': 2303, 'b1f1': 2304, 'b1g1': 2305, 'b1h1': 2306, 'b1b3': 2309, 'b1b4': 2310, 'b1b5': 2311, 'b1b6': 2312, 
#     'b1b7': 2313, 'b1b8': 2314, 'c1e1': 2323, 'c1f1': 2324, 'c1g1': 2325, 'c1h1': 2326, 'c1a1': 2328, 'c1c3': 2330, 'c1c4': 2331, 'c1c5': 2332, 
#     'c1c6': 2333, 'c1c7': 2334, 'c1c8': 2335, 'd1f1': 2344, 'd1g1': 2345, 'd1h1': 2346, 'd1b1': 2348, 'd1a1': 2349, 'd1d3': 2351, 'd1d4': 2352, 
#     'd1d5': 2353, 'd1d6': 2354, 'd1d7': 2355, 'd1d8': 2356, 'e1h1': 2366, 'e1b1': 2369, 'e1a1': 2370, 'e1e3': 2372, 'e1e4': 2373, 'e1e5': 2374, 
#     'e1e6': 2375, 'e1e7': 2376, 'e1e8': 2377, 'f1h1': 2386, 'f1d1': 2388, 'f1c1': 2389, 'f1b1': 2390, 'f1a1': 2391, 'f1f3': 2393, 'f1f4': 2394, 
#     'f1f5': 2395, 'f1f6': 2396, 'f1f7': 2397, 'f1f8': 2398, 'g1e1': 2408, 'g1d1': 2409, 'g1c1': 2410, 'g1b1': 2411, 'g1a1': 2412, 'g1g3': 2414, 
#     'g1g4': 2415, 'g1g5': 2416, 'g1g6': 2417, 'g1g7': 2418, 'g1g8': 2419, 'h1f1': 2428, 'h1e1': 2429, 'h1d1': 2430, 'h1c1': 2431, 'h1b1': 2432, 
#     'h1a1': 2433, 'h1h3': 2435, 'h1h4': 2436, 'h1h5': 2437, 'h1h6': 2438, 'h1h7': 2439, 'h1h8': 2440, 'a2c2': 2449, 'a2d2': 2450, 'a2e2': 2451, 
#     'a2f2': 2452, 'a2g2': 2453, 'a2h2': 2454, 'a2a5': 2457, 'a2a6': 2458, 'a2a7': 2459, 'a2a8': 2460, 'b2d2': 2470, 'b2e2': 2471, 'b2f2': 2472, 
#     'b2g2': 2473, 'b2h2': 2474, 'b2b5': 2478, 'b2b6': 2479, 'b2b7': 2480, 'b2b8': 2481, 'c2e2': 2493, 'c2f2': 2494, 'c2g2': 2495, 'c2h2': 2496, 
#     'c2a2': 2498, 'c2c5': 2501, 'c2c6': 2502, 'c2c7': 2503, 'c2c8': 2504, 'd2f2': 2516, 'd2g2': 2517, 'd2h2': 2518, 'd2b2': 2520, 'd2a2': 2521, 
#     'd2d5': 2524, 'd2d6': 2525, 'd2d7': 2526, 'd2d8': 2527, 'e2g2': 2539, 'e2h2': 2540, 'e2c2': 2542, 'e2b2': 2543, 'e2a2': 2544, 'e2e5': 2547, 
#     'e2e6': 2548, 'e2e7': 2549, 'e2e8': 2550, 'f2h2': 2562, 'f2d2': 2564, 'f2c2': 2565, 'f2b2': 2566, 'f2a2': 2567, 'f2f5': 2570, 'f2f6': 2571, 
#     'f2f7': 2572, 'f2f8': 2573, 'g2e2': 2586, 'g2d2': 2587, 'g2c2': 2588, 'g2b2': 2589, 'g2a2': 2590, 'g2g5': 2593, 'g2g6': 2594, 'g2g7': 2595, 
#     'g2g8': 2596, 'h2f2': 2608, 'h2e2': 2609, 'h2d2': 2610, 'h2c2': 2611, 'h2b2': 2612, 'h2a2': 2613, 'h2h5': 2616, 'h2h6': 2617, 'h2h7': 2618, 
#     'h2h8': 2619, 'a3c3': 2629, 'a3d3': 2630, 'a3e3': 2631, 'a3f3': 2632, 'a3g3': 2633, 'a3h3': 2634, 'a3a5': 2636, 'a3a6': 2637, 'a3a7': 2638, 
#     'a3a8': 2639, 'a3a1': 2641, 'b3d3': 2650, 'b3e3': 2651, 'b3f3': 2652, 'b3g3': 2653, 'b3h3': 2654, 'b3b5': 2657, 'b3b6': 2658, 'b3b7': 2659, 
#     'b3b8': 2660, 'b3b1': 2662, 'c3e3': 2673, 'c3f3': 2674, 'c3g3': 2675, 'c3h3': 2676, 'c3a3': 2678, 'c3c5': 2680, 'c3c6': 2681, 'c3c7': 2682, 
#     'c3c8': 2683, 'c3c1': 2685, 'd3f3': 2698, 'd3g3': 2699, 'd3h3': 2700, 'd3b3': 2702, 'd3a3': 2703, 'd3d5': 2705, 'd3d6': 2706, 'd3d7': 2707, 
#     'd3d8': 2708, 'd3d1': 2710, 'e3g3': 2723, 'e3h3': 2724, 'e3c3': 2726, 'e3b3': 2727, 'e3a3': 2728, 'e3e5': 2730, 'e3e6': 2731, 'e3e7': 2732, 
#     'e3e8': 2733, 'e3e1': 2735, 'f3h3': 2748, 'f3d3': 2750, 'f3c3': 2751, 'f3b3': 2752, 'f3a3': 2753, 'f3f5': 2755, 'f3f6': 2756, 'f3f7': 2757, 
#     'f3f8': 2758, 'f3f1': 2760, 'g3e3': 2774, 'g3d3': 2775, 'g3c3': 2776, 'g3b3': 2777, 'g3a3': 2778, 'g3g5': 2780, 'g3g6': 2781, 'g3g7': 2782, 
#     'g3g8': 2783, 'g3g1': 2785, 'h3f3': 2796, 'h3e3': 2797, 'h3d3': 2798, 'h3c3': 2799, 'h3b3': 2800, 'h3a3': 2801, 'h3h5': 2803, 'h3h6': 2804, 
#     'h3h7': 2805, 'h3h8': 2806, 'h3h1': 2808, 'a4c4': 2817, 'a4d4': 2818, 'a4e4': 2819, 'a4f4': 2820, 'a4g4': 2821, 'a4h4': 2822, 'a4a6': 2824, 
#     'a4a7': 2825, 'a4a8': 2826, 'a4a2': 2828, 'a4a1': 2829, 'b4d4': 2838, 'b4e4': 2839, 'b4f4': 2840, 'b4g4': 2841, 'b4h4': 2842, 'b4b6': 2845, 
#     'b4b7': 2846, 'b4b8': 2847, 'b4b2': 2849, 'b4b1': 2850, 'c4e4': 2861, 'c4f4': 2862, 'c4g4': 2863, 'c4h4': 2864, 'c4a4': 2866, 'c4c6': 2868, 
#     'c4c7': 2869, 'c4c8': 2870, 'c4c2': 2872, 'c4c1': 2873, 'd4f4': 2886, 'd4g4': 2887, 'd4h4': 2888, 'd4b4': 2890, 'd4a4': 2891, 'd4d6': 2893, 
#     'd4d7': 2894, 'd4d8': 2895, 'd4d2': 2897, 'd4d1': 2898, 'e4g4': 2913, 'e4h4': 2914, 'e4c4': 2916, 'e4b4': 2917, 'e4a4': 2918, 'e4e6': 2920, 
#     'e4e7': 2921, 'e4e8': 2922, 'e4e2': 2924, 'e4e1': 2925, 'f4h4': 2940, 'f4d4': 2942, 'f4c4': 2943, 'f4b4': 2944, 'f4a4': 2945, 'f4f6': 2947, 
#     'f4f7': 2948, 'f4f8': 2949, 'f4f2': 2951, 'f4f1': 2952, 'g4e4': 2966, 'g4d4': 2967, 'g4c4': 2968, 'g4b4': 2969, 'g4a4': 2970, 'g4g6': 2972, 
#     'g4g7': 2973, 'g4g8': 2974, 'g4g2': 2976, 'g4g1': 2977, 'h4f4': 2988, 'h4e4': 2989, 'h4d4': 2990, 'h4c4': 2991, 'h4b4': 2992, 'h4a4': 2993, 
#     'h4h6': 2995, 'h4h7': 2996, 'h4h8': 2997, 'h4h2': 2999, 'h4h1': 3000, 'a5c5': 3009, 'a5d5': 3010, 'a5e5': 3011, 'a5f5': 3012, 'a5g5': 3013, 
#     'a5h5': 3014, 'a5a7': 3016, 'a5a8': 3017, 'a5a3': 3019, 'a5a2': 3020, 'a5a1': 3021, 'b5d5': 3030, 'b5e5': 3031, 'b5f5': 3032, 'b5g5': 3033, 
#     'b5h5': 3034, 'b5b7': 3037, 'b5b8': 3038, 'b5b3': 3040, 'b5b2': 3041, 'b5b1': 3042, 'c5e5': 3053, 'c5f5': 3054, 'c5g5': 3055, 'c5h5': 3056, 
#     'c5a5': 3058, 'c5c7': 3060, 'c5c8': 3061, 'c5c3': 3063, 'c5c2': 3064, 'c5c1': 3065, 'd5f5': 3078, 'd5g5': 3079, 'd5h5': 3080, 'd5b5': 3082, 
#     'd5a5': 3083, 'd5d7': 3085, 'd5d8': 3086, 'd5d3': 3088, 'd5d2': 3089, 'd5d1': 3090, 'e5g5': 3105, 'e5h5': 3106, 'e5c5': 3108, 'e5b5': 3109, 
#     'e5a5': 3110, 'e5e7': 3112, 'e5e8': 3113, 'e5e3': 3115, 'e5e2': 3116, 'e5e1': 3117, 'f5h5': 3132, 'f5d5': 3134, 'f5c5': 3135, 'f5b5': 3136, 
#     'f5a5': 3137, 'f5f7': 3139, 'f5f8': 3140, 'f5f3': 3142, 'f5f2': 3143, 'f5f1': 3144, 'g5e5': 3158, 'g5d5': 3159, 'g5c5': 3160, 'g5b5': 3161, 
#     'g5a5': 3162, 'g5g7': 3164, 'g5g8': 3165, 'g5g3': 3167, 'g5g2': 3168, 'g5g1': 3169, 'h5f5': 3180, 'h5e5': 3181, 'h5d5': 3182, 'h5c5': 3183, 
#     'h5b5': 3184, 'h5a5': 3185, 'h5h7': 3187, 'h5h8': 3188, 'h5h3': 3190, 'h5h2': 3191, 'h5h1': 3192, 'a6c6': 3201, 'a6d6': 3202, 'a6e6': 3203, 
#     'a6f6': 3204, 'a6g6': 3205, 'a6h6': 3206, 'a6a8': 3208, 'a6a4': 3210, 'a6a3': 3211, 'a6a2': 3212, 'a6a1': 3213, 'b6d6': 3222, 'b6e6': 3223, 
#     'b6f6': 3224, 'b6g6': 3225, 'b6h6': 3226, 'b6b8': 3229, 'b6b4': 3231, 'b6b3': 3232, 'b6b2': 3233, 'b6b1': 3234, 'c6e6': 3245, 'c6f6': 3246, 
#     'c6g6': 3247, 'c6h6': 3248, 'c6a6': 3250, 'c6c8': 3252, 'c6c4': 3254, 'c6c3': 3255, 'c6c2': 3256, 'c6c1': 3257, 'd6f6': 3270, 'd6g6': 3271, 
#     'd6h6': 3272, 'd6b6': 3274, 'd6a6': 3275, 'd6d8': 3277, 'd6d4': 3279, 'd6d3': 3280, 'd6d2': 3281, 'd6d1': 3282, 'e6g6': 3295, 'e6h6': 3296, 
#     'e6c6': 3298, 'e6b6': 3299, 'e6a6': 3300, 'e6e8': 3302, 'e6e4': 3304, 'e6e3': 3305, 'e6e2': 3306, 'e6e1': 3307, 'f6h6': 3320, 'f6d6': 3322, 
#     'f6c6': 3323, 'f6b6': 3324, 'f6a6': 3325, 'f6f8': 3327, 'f6f4': 3329, 'f6f3': 3330, 'f6f2': 3331, 'f6f1': 3332, 'g6e6': 3346, 'g6d6': 3347, 
#     'g6c6': 3348, 'g6b6': 3349, 'g6a6': 3350, 'g6g8': 3352, 'g6g4': 3354, 'g6g3': 3355, 'g6g2': 3356, 'g6g1': 3357, 'h6f6': 3368, 'h6e6': 3369, 
#     'h6d6': 3370, 'h6c6': 3371, 'h6b6': 3372, 'h6a6': 3373, 'h6h8': 3375, 'h6h4': 3377, 'h6h3': 3378, 'h6h2': 3379, 'h6h1': 3380, 'a7c7': 3389, 
#     'a7d7': 3390, 'a7e7': 3391, 'a7f7': 3392, 'a7g7': 3393, 'a7h7': 3394, 'a7a5': 3397, 'a7a4': 3398, 'a7a3': 3399, 'a7a2': 3400, 'a7a1': 3401, 
#     'b7d7': 3410, 'b7e7': 3411, 'b7f7': 3412, 'b7g7': 3413, 'b7h7': 3414, 'b7b5': 3418, 'b7b4': 3419, 'b7b3': 3420, 'b7b2': 3421, 'b7b1': 3422, 
#     'c7e7': 3433, 'c7f7': 3434, 'c7g7': 3435, 'c7h7': 3436, 'c7a7': 3438, 'c7c5': 3441, 'c7c4': 3442, 'c7c3': 3443, 'c7c2': 3444, 'c7c1': 3445, 
#     'd7f7': 3456, 'd7g7': 3457, 'd7h7': 3458, 'd7b7': 3460, 'd7a7': 3461, 'd7d5': 3464, 'd7d4': 3465, 'd7d3': 3466, 'd7d2': 3467, 'd7d1': 3468, 
#     'e7g7': 3479, 'e7h7': 3480, 'e7c7': 3482, 'e7b7': 3483, 'e7a7': 3484, 'e7e5': 3487, 'e7e4': 3488, 'e7e3': 3489, 'e7e2': 3490, 'e7e1': 3491, 
#     'f7h7': 3502, 'f7d7': 3504, 'f7c7': 3505, 'f7b7': 3506, 'f7a7': 3507, 'f7f5': 3510, 'f7f4': 3511, 'f7f3': 3512, 'f7f2': 3513, 'f7f1': 3514, 
#     'g7e7': 3526, 'g7d7': 3527, 'g7c7': 3528, 'g7b7': 3529, 'g7a7': 3530, 'g7g5': 3533, 'g7g4': 3534, 'g7g3': 3535, 'g7g2': 3536, 'g7g1': 3537, 
#     'h7f7': 3548, 'h7e7': 3549, 'h7d7': 3550, 'h7c7': 3551, 'h7b7': 3552, 'h7a7': 3553, 'h7h5': 3556, 'h7h4': 3557, 'h7h3': 3558, 'h7h2': 3559, 
#     'h7h1': 3560, 'a8c8': 3569, 'a8d8': 3570, 'a8e8': 3571, 'a8f8': 3572, 'a8g8': 3573, 'a8h8': 3574, 'a8a6': 3576, 'a8a5': 3577, 'a8a4': 3578, 
#     'a8a3': 3579, 'a8a2': 3580, 'a8a1': 3581, 'b8d8': 3590, 'b8e8': 3591, 'b8f8': 3592, 'b8g8': 3593, 'b8h8': 3594, 'b8b6': 3597, 'b8b5': 3598, 
#     'b8b4': 3599, 'b8b3': 3600, 'b8b2': 3601, 'b8b1': 3602, 'c8e8': 3611, 'c8f8': 3612, 'c8g8': 3613, 'c8h8': 3614, 'c8a8': 3616, 'c8c6': 3618, 
#     'c8c5': 3619, 'c8c4': 3620, 'c8c3': 3621, 'c8c2': 3622, 'c8c1': 3623, 'd8f8': 3632, 'd8g8': 3633, 'd8h8': 3634, 'd8b8': 3636, 'd8a8': 3637, 
#     'd8d6': 3639, 'd8d5': 3640, 'd8d4': 3641, 'd8d3': 3642, 'd8d2': 3643, 'd8d1': 3644, 'e8h8': 3654, 'e8b8': 3657, 'e8a8': 3658, 'e8e6': 3660, 
#     'e8e5': 3661, 'e8e4': 3662, 'e8e3': 3663, 'e8e2': 3664, 'e8e1': 3665, 'f8h8': 3674, 'f8d8': 3676, 'f8c8': 3677, 'f8b8': 3678, 'f8a8': 3679, 
#     'f8f6': 3681, 'f8f5': 3682, 'f8f4': 3683, 'f8f3': 3684, 'f8f2': 3685, 'f8f1': 3686, 'g8e8': 3696, 'g8d8': 3697, 'g8c8': 3698, 'g8b8': 3699, 
#     'g8a8': 3700, 'g8g6': 3702, 'g8g5': 3703, 'g8g4': 3704, 'g8g3': 3705, 'g8g2': 3706, 'g8g1': 3707, 'h8f8': 3716, 'h8e8': 3717, 'h8d8': 3718, 
#     'h8c8': 3719, 'h8b8': 3720, 'h8a8': 3721, 'h8h6': 3723, 'h8h5': 3724, 'h8h4': 3725, 'h8h3': 3726, 'h8h2': 3727, 'h8h1': 3728, 'a1c3': 2295, 
#     'a1d4': 2296, 'a1e5': 2297, 'a1f6': 2298, 'a1g7': 2299, 'a1h8': 2300, 'b1d3': 2316, 'b1e4': 2317, 'b1f5': 2318, 'b1g6': 2319, 'b1h7': 2320, 
#     'c1e3': 2337, 'c1f4': 2338, 'c1g5': 2339, 'c1h6': 2340, 'c1a3': 2342, 'd1f3': 2358, 'd1g4': 2359, 'd1h5': 2360, 'd1b3': 2362, 'd1a4': 2363, 
#     'e1g3': 2379, 'e1h4': 2380, 'e1c3': 2382, 'e1b4': 2383, 'e1a5': 2384, 'f1h3': 2400, 'f1d3': 2402, 'f1c4': 2403, 'f1b5': 2404, 'f1a6': 2405, 
#     'g1e3': 2422, 'g1d4': 2423, 'g1c5': 2424, 'g1b6': 2425, 'g1a7': 2426, 'h1f3': 2442, 'h1e4': 2443, 'h1d5': 2444, 'h1c6': 2445, 'h1b7': 2446, 
#     'h1a8': 2447, 'a2c4': 2463, 'a2d5': 2464, 'a2e6': 2465, 'a2f7': 2466, 'a2g8': 2467, 'b2d4': 2484, 'b2e5': 2485, 'b2f6': 2486, 'b2g7': 2487, 
#     'b2h8': 2488, 'c2e4': 2507, 'c2f5': 2508, 'c2g6': 2509, 'c2h7': 2510, 'c2a4': 2513, 'd2f4': 2530, 'd2g5': 2531, 'd2h6': 2532, 'd2b4': 2535, 
#     'd2a5': 2536, 'e2g4': 2553, 'e2h5': 2554, 'e2c4': 2557, 'e2b5': 2558, 'e2a6': 2559, 'f2h4': 2576, 'f2d4': 2579, 'f2c5': 2580, 'f2b6': 2581, 
#     'f2a7': 2582, 'g2e4': 2601, 'g2d5': 2602, 'g2c6': 2603, 'g2b7': 2604, 'g2a8': 2605, 'h2f4': 2622, 'h2e5': 2623, 'h2d6': 2624, 'h2c7': 2625, 
#     'h2b8': 2626, 'a3c5': 2643, 'a3d6': 2644, 'a3e7': 2645, 'a3f8': 2646, 'a3c1': 2648, 'b3d5': 2664, 'b3e6': 2665, 'b3f7': 2666, 'b3g8': 2667, 
#     'b3d1': 2669, 'c3e5': 2687, 'c3f6': 2688, 'c3g7': 2689, 'c3h8': 2690, 'c3e1': 2692, 'c3a5': 2694, 'c3a1': 2696, 'd3f5': 2712, 'd3g6': 2713, 
#     'd3h7': 2714, 'd3f1': 2716, 'd3b5': 2718, 'd3a6': 2719, 'd3b1': 2721, 'e3g5': 2737, 'e3h6': 2738, 'e3g1': 2740, 'e3c5': 2742, 'e3b6': 2743, 
#     'e3a7': 2744, 'e3c1': 2746, 'f3h5': 2762, 'f3h1': 2764, 'f3d5': 2766, 'f3c6': 2767, 'f3b7': 2768, 'f3a8': 2769, 'f3d1': 2771, 'g3e5': 2789, 
#     'g3d6': 2790, 'g3c7': 2791, 'g3b8': 2792, 'g3e1': 2794, 'h3f5': 2810, 'h3e6': 2811, 'h3d7': 2812, 'h3c8': 2813, 'h3f1': 2815, 'a4c6': 2831, 
#     'a4d7': 2832, 'a4e8': 2833, 'a4c2': 2835, 'a4d1': 2836, 'b4d6': 2852, 'b4e7': 2853, 'b4f8': 2854, 'b4d2': 2856, 'b4e1': 2857, 'c4e6': 2875, 
#     'c4f7': 2876, 'c4g8': 2877, 'c4e2': 2879, 'c4f1': 2880, 'c4a6': 2882, 'c4a2': 2884, 'd4f6': 2900, 'd4g7': 2901, 'd4h8': 2902, 'd4f2': 2904, 
#     'd4g1': 2905, 'd4b6': 2907, 'd4a7': 2908, 'd4b2': 2910, 'd4a1': 2911, 'e4g6': 2927, 'e4h7': 2928, 'e4g2': 2930, 'e4h1': 2931, 'e4c6': 2933, 
#     'e4b7': 2934, 'e4a8': 2935, 'e4c2': 2937, 'e4b1': 2938, 'f4h6': 2954, 'f4h2': 2956, 'f4d6': 2958, 'f4c7': 2959, 'f4b8': 2960, 'f4d2': 2962, 
#     'f4c1': 2963, 'g4e6': 2981, 'g4d7': 2982, 'g4c8': 2983, 'g4e2': 2985, 'g4d1': 2986, 'h4f6': 3002, 'h4e7': 3003, 'h4d8': 3004, 'h4f2': 3006, 
#     'h4e1': 3007, 'a5c7': 3023, 'a5d8': 3024, 'a5c3': 3026, 'a5d2': 3027, 'a5e1': 3028, 'b5d7': 3044, 'b5e8': 3045, 'b5d3': 3047, 'b5e2': 3048, 
#     'b5f1': 3049, 'c5e7': 3067, 'c5f8': 3068, 'c5e3': 3070, 'c5f2': 3071, 'c5g1': 3072, 'c5a7': 3074, 'c5a3': 3076, 'd5f7': 3092, 'd5g8': 3093, 
#     'd5f3': 3095, 'd5g2': 3096, 'd5h1': 3097, 'd5b7': 3099, 'd5a8': 3100, 'd5b3': 3102, 'd5a2': 3103, 'e5g7': 3119, 'e5h8': 3120, 'e5g3': 3122, 
#     'e5h2': 3123, 'e5c7': 3125, 'e5b8': 3126, 'e5c3': 3128, 'e5b2': 3129, 'e5a1': 3130, 'f5h7': 3146, 'f5h3': 3148, 'f5d7': 3150, 'f5c8': 3151, 
#     'f5d3': 3153, 'f5c2': 3154, 'f5b1': 3155, 'g5e7': 3173, 'g5d8': 3174, 'g5e3': 3176, 'g5d2': 3177, 'g5c1': 3178, 'h5f7': 3194, 'h5e8': 3195, 
#     'h5f3': 3197, 'h5e2': 3198, 'h5d1': 3199, 'a6c8': 3215, 'a6c4': 3217, 'a6d3': 3218, 'a6e2': 3219, 'a6f1': 3220, 'b6d8': 3236, 'b6d4': 3238, 
#     'b6e3': 3239, 'b6f2': 3240, 'b6g1': 3241, 'c6e8': 3259, 'c6e4': 3261, 'c6f3': 3262, 'c6g2': 3263, 'c6h1': 3264, 'c6a8': 3266, 'c6a4': 3268, 
#     'd6f8': 3284, 'd6f4': 3286, 'd6g3': 3287, 'd6h2': 3288, 'd6b8': 3290, 'd6b4': 3292, 'd6a3': 3293, 'e6g8': 3309, 'e6g4': 3311, 'e6h3': 3312, 
#     'e6c8': 3314, 'e6c4': 3316, 'e6b3': 3317, 'e6a2': 3318, 'f6h8': 3334, 'f6h4': 3336, 'f6d8': 3338, 'f6d4': 3340, 'f6c3': 3341, 'f6b2': 3342, 
#     'f6a1': 3343, 'g6e8': 3361, 'g6e4': 3363, 'g6d3': 3364, 'g6c2': 3365, 'g6b1': 3366, 'h6f8': 3382, 'h6f4': 3384, 'h6e3': 3385, 'h6d2': 3386, 
#     'h6c1': 3387, 'a7c5': 3404, 'a7d4': 3405, 'a7e3': 3406, 'a7f2': 3407, 'a7g1': 3408, 'b7d5': 3425, 'b7e4': 3426, 'b7f3': 3427, 'b7g2': 3428, 
#     'b7h1': 3429, 'c7e5': 3448, 'c7f4': 3449, 'c7g3': 3450, 'c7h2': 3451, 'c7a5': 3454, 'd7f5': 3471, 'd7g4': 3472, 'd7h3': 3473, 'd7b5': 3476, 
#     'd7a4': 3477, 'e7g5': 3494, 'e7h4': 3495, 'e7c5': 3498, 'e7b4': 3499, 'e7a3': 3500, 'f7h5': 3517, 'f7d5': 3520, 'f7c4': 3521, 'f7b3': 3522, 
#     'f7a2': 3523, 'g7e5': 3542, 'g7d4': 3543, 'g7c3': 3544, 'g7b2': 3545, 'g7a1': 3546, 'h7f5': 3563, 'h7e4': 3564, 'h7d3': 3565, 'h7c2': 3566, 
#     'h7b1': 3567, 'a8c6': 3583, 'a8d5': 3584, 'a8e4': 3585, 'a8f3': 3586, 'a8g2': 3587, 'a8h1': 3588, 'b8d6': 3604, 'b8e5': 3605, 'b8f4': 3606, 
#     'b8g3': 3607, 'b8h2': 3608, 'c8e6': 3625, 'c8f5': 3626, 'c8g4': 3627, 'c8h3': 3628, 'c8a6': 3630, 'd8f6': 3646, 'd8g5': 3647, 'd8h4': 3648, 
#     'd8b6': 3650, 'd8a5': 3651, 'e8g6': 3667, 'e8h5': 3668, 'e8c6': 3670, 'e8b5': 3671, 'e8a4': 3672, 'f8h6': 3688, 'f8d6': 3690, 'f8c5': 3691, 
#     'f8b4': 3692, 'f8a3': 3693, 'g8e6': 3710, 'g8d5': 3711, 'g8c4': 3712, 'g8b3': 3713, 'g8a2': 3714, 'h8f6': 3730, 'h8e5': 3731, 'h8d4': 3732, 
#     'h8c3': 3733, 'h8b2': 3734, 'h8a1': 3735, 'd2e1b': 3736, 'a2a1q': 3737, 'd2e1q': 3738, 'e2f1b': 3739, 'b7a8r': 3740, 'a2b1b': 3741, 'g2g1n': 3742, 
#     'a2b1q': 3743, 'd7e8n': 3744, 'e2e1r': 3745, 'd2d1r': 3746, 'a7b8b': 3747, 'd2c1q': 3748, 'd7e8r': 3749, 'h2h1r': 3750, 'f2e1r': 3751, 'b2a1n': 3752, 
#     'e7d8b': 3753, 'd7c8n': 3754, 'c7d8b': 3755, 'c7d8r': 3756, 'e2d1n': 3757, 'c2b1q': 3758, 'e7f8b': 3759, 'a2b1n': 3760, 'b2a1q': 3761, 'c2b1b': 3762, 
#     'b2c1b': 3763, 'e2e1n': 3764, 'b2a1b': 3765, 'b7c8r': 3766, 'h2h1q': 3767, 'a7b8q': 3768, 'c7b8q': 3769, 'f2g1r': 3770, 'g2f1r': 3771, 'g7f8q': 3772, 
#     'e2e1b': 3773, 'h2h1n': 3774, 'c7b8b': 3775, 'b2c1q': 3776, 'f2g1n': 3777, 'g2f1b': 3778, 'a2a1b': 3779, 'f7e8q': 3780, 'e7f8n': 3781, 'e7d8q': 3782, 
#     'e2d1b': 3783, 'c7b8r': 3784, 'b7c8n': 3785, 'f7e8b': 3786, 'e2f1n': 3787, 'c2c1n': 3788, 'c2c1q': 3789, 'h2g1n': 3790, 'd7e8q': 3791, 'c2c1b': 3792, 
#     'b2b1q': 3793, 'd7c8b': 3794, 'd2c1r': 3795, 'b2b1n': 3796, 'b2b1r': 3797, 'b7a8q': 3798, 'h2g1q': 3799, 'f2f1b': 3800, 'a2a1n': 3801, 'e2d1r': 3802, 
#     'd2d1q': 3803, 'd2e1r': 3804, 'd7c8q': 3805, 'f2f1q': 3806, 'g2f1n': 3807, 'g2h1n': 3808, 'e2f1q': 3809, 'g7h8n': 3810, 'b2c1r': 3811, 'a7b8n': 3812, 
#     'f2g1b': 3813, 'e2d1q': 3814, 'g2h1q': 3815, 'g2g1b': 3816, 'h2h1b': 3817, 'b2b1b': 3818, 'h7g8b': 3819, 'c2b1n': 3820, 'a7b8r': 3821, 'f7e8r': 3822, 
#     'g2g1r': 3823, 'f7g8r': 3824, 'h7g8r': 3825, 'g2h1r': 3826, 'e2f1r': 3827, 'c7b8n': 3828, 'f2g1q': 3829, 'e7f8q': 3830, 'e2e1q': 3831, 'd2d1n': 3832, 
#     'c7d8n': 3833, 'h7g8q': 3834, 'e7f8r': 3835, 'c2d1n': 3836, 'd2e1n': 3837, 'f2f1n': 3838, 'h2g1b': 3839, 'f2f1r': 3840, 'd2d1b': 3841, 'h7g8n': 3842, 
#     'd7c8r': 3843, 'c2c1r': 3844, 'b7c8b': 3845, 'h2g1r': 3846, 'g7h8b': 3847, 'f2e1q': 3848, 'a2b1r': 3849, 'g7f8r': 3850, 'g2g1q': 3851, 'c2d1q': 3852, 
#     'c7d8q': 3853, 'f2e1n': 3854, 'b2c1n': 3855, 'b7a8n': 3856, 'e7d8n': 3857, 'g7h8q': 3858, 'b7c8q': 3859, 'b7a8b': 3860, 'g7f8n': 3861, 'g2h1b': 3862, 
#     'g2f1q': 3863, 'f2e1b': 3864, 'f7g8b': 3865, 'a2a1r': 3866, 'c2d1r': 3867, 'd2c1n': 3868, 'g7h8r': 3869, 'c2b1r': 3870, 'e7d8r': 3871, 'f7g8n': 3872, 
#     'g7f8b': 3873, 'b2a1r': 3874, 'f7e8n': 3875, 'd2c1b': 3876, 'c2d1b': 3877, 'd7e8b': 3878, 'f7g8q': 3879}


def compact_mapping(original_mapping):
    """
    Reassigns values of the original mapping to sequential integers without gaps.

    Args:
        original_mapping (dict): The current mapping of moves to indices.

    Returns:
        dict: A new mapping with sequential indices starting from 0.
    """
    # Sort the original mapping by its values to maintain order
    sorted_items = sorted(original_mapping.items(), key=lambda x: x[1])

    # Create a new mapping with sequential indices
    new_mapping = {}
    new_index = 0

    for move, _ in sorted_items:
        new_mapping[move] = new_index
        new_index += 1

    return new_mapping


new_mapping = compact_mapping(mapping)
