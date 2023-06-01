import subprocess
import numpy as np

def get_time1(M, N, BLOCK_X, TILE_SIZE, filename):
    # subprocess.run(["nvcc", filename+".cu", "-o", filename])

    command = ["./"+filename, str(M), str(N), str(BLOCK_X), str(TILE_SIZE)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    output, error = process.communicate()

    output = output.decode("utf-8")
    error = error.decode("utf-8")
    print(M, N, BLOCK_X, TILE_SIZE, output)

    # latency = float(output.strip().split("\n")[0].split()[1])
    latency = 0.0
    output = output.strip().split("\n")[0].split()
    if len(output) > 1:
        latency = float(output[1])
    # print("is_match:", is_match)
    # print("Latency:", latency)
    return latency


def profile_schedule1(file):
    # results = {}
    # blocksz =  [32, 64, 128, 256, 512, 1024]
    # tile = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

    # for M in range(512, 8192 + 1, 512):
    #     for N in range(512, 8192 + 1, 512):
    #         results[(M, N)] = {}  
    #         for BLOCK_X in blocksz:
    #             for TILE_SIZE in tile:
    #                 if TILE_SIZE > M or BLOCK_X > N:
    #                     continue
    #                 latency = get_time1(M, N, BLOCK_X, TILE_SIZE, file)
    #                 if not latency:
    #                     continue 
    #                 if float(latency) > 0.000001:
    #                     results[(M, N)][(BLOCK_X, TILE_SIZE)] = latency
    # print("Results", results)
    # np.save("reduction1_results.npy", results)

    results = np.load("reduction1_results.npy", allow_pickle=True).item()
    candidates = {}
    # top_result = {}
    for (M, N), block_results in results.items():
        sorted_results = sorted(block_results.items(), key=lambda x: x[1])
        # top_result[(M, N)] = sorted_results[0]
        top_result = sorted_results[0]
        # print("Top result for (M, N) =", top_result)
        print("Top result for (M, N) =", (M, N), ":", top_result)
        for (BLOCK_X, TILE_SIZE), latency in sorted_results:
            if len(top_result) > 1 and float(latency) <= float(top_result[1]) / 0.95:
                if (BLOCK_X, TILE_SIZE) not in candidates:
                    candidates[(BLOCK_X, TILE_SIZE)] = 1
                else:
                    candidates[(BLOCK_X, TILE_SIZE)] += 1
    
    np.save("reduction1_candidates.npy", candidates)
    # np.save("reduction1_top_results.npy", top_result)
    best_candidate = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    print("Best candidates for schedule 1:")
    for (BLOCK_X, TILE_SIZE), count in best_candidate:
        print(f"BLOCK_X={BLOCK_X}, TILE_SIZE={TILE_SIZE}: Count={count}")
    
    # Save the results to a file
    # np.save("reduction1_results.npy", results)
    # np.save("reduction1_results.npy", best_candidate)
    # np.save("column_reduction_results.npy", sorted_results)
def profile_candidates(file):
    results = {}
    candidates = [
        (64, 32),
        (256, 32),
        (512, 32),
        (128, 32),
        (128, 64),
        (256, 64),
        (32, 32),
        (64, 64),
        (512, 64),
        (1024, 32),
        (32, 64),
        (1024, 64),
        (256, 128),
        (512, 128),
        (128, 128),
        (64, 128)
    ]

    total_results = np.load("reduction1_results.npy", allow_pickle=True).item()

    for N in range(512, 8192 + 1, 512):
        for M in range(512, 8192 + 1, 512):
            results[(M, N)] = {}  
            for (BLOCK_X, TILE_SIZE) in candidates:
                if TILE_SIZE > M or BLOCK_X > N:
                    continue
                results[(M, N)][(BLOCK_X, TILE_SIZE)] = total_results[(M, N)][(BLOCK_X, TILE_SIZE)]
                # latency = get_time1(M, N, BLOCK_X, TILE_SIZE, file)
                # if latency > 0.00001:
                    # results[(M, N)][(BLOCK_X, TILE_SIZE)] = latency

    # print("Results", results)
    np.save("reduction1_results2.npy", results)
    # results = np.load("reduction2_results.npy", allow_pickle=True).item()
    hit_times = {}
    hit_shapes = {}
    top_results = np.load("reduction1_top_results.npy", allow_pickle=True).item()
    for (M, N), block_results in results.items():
        sorted_results = sorted(block_results.items(), key=lambda x: x[1])
        # print("Sorted results:", sorted_results)
        if len(sorted_results[0]) < 2:
            continue
        top_result = float(sorted_results[0][1])
        # top_result = top_results[(M,N)][0][2]
        # if top_result / 0.95 < float(sorted_results[0][1]):
            # top_result = float(sorted_results[0][1]) * 0.9
        # print("Top result for (M, N) =", (M, N), ":", top_results)
        for (BLOCK_X, TILE_SIZE), latency in sorted_results:
            if latency < top_result / 0.95:
                if (BLOCK_X, TILE_SIZE) not in hit_times:
                    hit_times[(BLOCK_X, TILE_SIZE)] = 1
                    hit_shapes[(BLOCK_X, TILE_SIZE)] = [(M, N)]
                else:
                    hit_times[(BLOCK_X, TILE_SIZE)] += 1
                    hit_shapes[(BLOCK_X, TILE_SIZE)].append((M, N))
    # hit_shapes = sorted(hit_shapes.items(), key=lambda x: x[1], reverse=True)
    
    np.save("reduction1_hit_times.npy", hit_times)
    np.save("reduction1_hit_shapes.npy", hit_shapes)
    hit_times = sorted(hit_times.items(), key=lambda x: x[1], reverse=True)
    print("Hit times", hit_times)
    print("Hit shapes", hit_shapes)
    # best_candidate = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    # print("Best candidates for schedule 2:")
    for (BLOCK_X, TILE_SIZE), count in hit_times:
        print(f"BLOCK_X={BLOCK_X}, TILE_SIZE={TILE_SIZE}: Count={count}")
    
# read top results from txt file 
def save_top_results():
    f = open("profile_reduct1_res.txt", "r")
    top_results = {}
    for line in f:
        line = line.strip()
        if line.startswith("Top result"):
            data = line.split("=")[1]
            shape, part = data.split(":")
            shape = shape.strip().strip("()")
            M, N = shape.split(",")
            M = int(M)
            N = int(N)
            part = part.strip().strip("()").strip(",")
            idx, time = part.split(")")
            blockx, tilesize = idx.split(",")
            blockx = int(blockx)
            tilesize = int(tilesize)
            time = time.strip(",")
            time = float(time)
            if idx not in top_results:
                top_results[(M, N)] = {}
                top_results[(M, N)] = [(blockx, tilesize, time)]

    f.close()
    np.save("reduction1_top_results.npy", top_results)

def profile_hit_shapes1(file):
    hit_times = np.load(file + "_hit_times.npy", allow_pickle=True).item()
    hit_shapes = np.load(file + "_hit_shapes.npy", allow_pickle=True).item()

    # get the count of each BLOCK_X
    hit_times_count_block_x = {}
    hit_times_count_tile = {}
    for (BLOCK_X, TILE_SIZE), count in hit_times.items():
        if BLOCK_X not in hit_times_count_block_x:
            hit_times_count_block_x[BLOCK_X] = count
        else:
            hit_times_count_block_x[BLOCK_X] += count
        if TILE_SIZE not in hit_times_count_tile:
            hit_times_count_tile[TILE_SIZE] = count
        else:
            hit_times_count_tile[TILE_SIZE] += count
    
    np.save(file + "_hit_times_count_block_x.npy", hit_times_count_block_x)
    np.save(file + "_hit_times_count_tile.npy", hit_times_count_tile)

    hit_times_count_block_x = sorted(hit_times_count_block_x.items(), key=lambda x: x[0])
    hit_times_count_tile = sorted(hit_times_count_tile.items(), key=lambda x: x[0])

    # get the range of M and N of every item in hit_shapes
    hit_shapes_range = {}
    for (BLOCK_X, TILE_SIZE), shapes in hit_shapes.items():
        shape_min = (8192, 8192)
        shape_max = (0, 0)
        for shape in shapes:
            M, N = shape
            if M < shape_min[0]:
                shape_min = (M, shape_min[1])
            if N < shape_min[1]:
                shape_min = (shape_min[0], N)
            if M > shape_max[0]:
                shape_max = (M, shape_max[1])
            if N > shape_max[1]:
                shape_max = (shape_max[0], N)
        hit_shapes_range[(BLOCK_X, TILE_SIZE)] = (shape_min, shape_max)
    # sort the hit_shapes_range by the number of times each (BLOCK_X, TILE_SIZE) is hit
    np.save(file + "_hit_shapes_range.npy", hit_shapes_range)

    # get how many times each M or N is hit for each (BLOCK_X, TILE_SIZE)
    hit_shapes_count_M = {}
    hit_shapes_count_N = {}
    for (BLOCK_X, TILE_SIZE), shapes in hit_shapes.items():
        hit_shapes_count_M[(BLOCK_X, TILE_SIZE)] = {}
        hit_shapes_count_N[(BLOCK_X, TILE_SIZE)] = {}
        for shape in shapes:
            M, N = shape
            if M not in hit_shapes_count_M[(BLOCK_X, TILE_SIZE)]:
                hit_shapes_count_M[(BLOCK_X, TILE_SIZE)][M] = 1
            else:
                hit_shapes_count_M[(BLOCK_X, TILE_SIZE)][M] += 1
            if N not in hit_shapes_count_N[(BLOCK_X, TILE_SIZE)]:
                hit_shapes_count_N[(BLOCK_X, TILE_SIZE)][N] = 1
            else:
                hit_shapes_count_N[(BLOCK_X, TILE_SIZE)][N] += 1
    np.save(file + "_hit_shapes_count_N.npy", hit_shapes_count_N)
    np.save(file + "_hit_shapes_count_M.npy", hit_shapes_count_M)

    hit_shapes_range = sorted(hit_shapes_range.items(), key=lambda x: x[0])
    hit_shapes_count_M = sorted(hit_shapes_count_M.items(), key=lambda x: x[0])
    hit_shapes_count_N = sorted(hit_shapes_count_N.items(), key=lambda x: x[0])
    
    print("Results for ", file, ":")

    for BLOCK_X, count in hit_times_count_block_x:
        print(f"BLOCK_X={BLOCK_X}: Count={count}")
    print("")
    for TILE_SIZE, count in hit_times_count_tile:
        print(f"TILE_SIZE={TILE_SIZE}: Count={count}")
    print("")

    for (BLOCK_X, TILE_SIZE), shapes in hit_shapes_range:
        print(f"BLOCK_X={BLOCK_X}, TILE_SIZE={TILE_SIZE}:")
        print(f"Min shape: {shapes[0]}")
        print(f"Max shape: {shapes[1]}")
        print("")

    for (BLOCK_X, TILE_SIZE), shapes in hit_shapes_count_M:
        print(f"BLOCK_X={BLOCK_X}, TILE_SIZE={TILE_SIZE}:")
        for M, count in shapes.items():
            print(f"M={M}: Count={count}")
        print("")

    for (BLOCK_X, TILE_SIZE), shapes in hit_shapes_count_N:
        print(f"BLOCK_X={BLOCK_X}, TILE_SIZE={TILE_SIZE}:")
        for N, count in shapes.items():
            print(f"N={N}: Count={count}")
        print("")   
    # print(file + " Hit shapes range", hit_shapes_range)
    # print in a more readable way



def profile_hit_shapes2(file):
    hit_times = np.load(file + "_hit_times.npy", allow_pickle=True).item()
    hit_shapes = np.load(file + "_hit_shapes.npy", allow_pickle=True).item()

    # get the count of each BLOCK_X
    hit_times_count_block_x = {}
    hit_times_count_block_y = {}
    hit_times_count_tile = {}
    for (BLOCK_X, BLOCK_Y, TILE_SIZE), count in hit_times.items():
        if BLOCK_X not in hit_times_count_block_x:
            hit_times_count_block_x[BLOCK_X] = count
        else:
            hit_times_count_block_x[BLOCK_X] += count
        if BLOCK_Y not in hit_times_count_block_y:
            hit_times_count_block_y[BLOCK_Y] = count
        else:
            hit_times_count_block_y[BLOCK_Y] += count
        if TILE_SIZE not in hit_times_count_tile:
            hit_times_count_tile[TILE_SIZE] = count
        else:
            hit_times_count_tile[TILE_SIZE] += count
    
    np.save(file + "_hit_times_count_block_x.npy", hit_times_count_block_x)
    np.save(file + "_hit_times_count_block_y.npy", hit_times_count_block_y)
    np.save(file + "_hit_times_count_tile.npy", hit_times_count_tile)

    hit_times_count_block_x = sorted(hit_times_count_block_x.items(), key=lambda x: x[0])
    hit_times_count_block_y = sorted(hit_times_count_block_y.items(), key=lambda x: x[0])
    hit_times_count_tile = sorted(hit_times_count_tile.items(), key=lambda x: x[0])


    # get the range of M and N of every item in hit_shapes
    hit_shapes_range = {}
    for (BLOCK_X, BLOCK_Y, TILE_SIZE), shapes in hit_shapes.items():
        shape_min = (8192, 8192)
        shape_max = (0, 0)
        for shape in shapes:
            M, N = shape
            if M < shape_min[0]:
                shape_min = (M, shape_min[1])
            if N < shape_min[1]:
                shape_min = (shape_min[0], N)
            if M > shape_max[0]:
                shape_max = (M, shape_max[1])
            if N > shape_max[1]:
                shape_max = (shape_max[0], N)
        hit_shapes_range[(BLOCK_X, BLOCK_Y, TILE_SIZE)] = (shape_min, shape_max)
    # sort the hit_shapes_range by the number of times each (BLOCK_X, TILE_SIZE) is hit
    np.save(file + "_hit_shapes_range.npy", hit_shapes_range)

    # get how many times each M or N is hit for each (BLOCK_X, TILE_SIZE)
    hit_shapes_count_M = {}
    hit_shapes_count_N = {}
    for (BLOCK_X, BLOCK_Y, TILE_SIZE), shapes in hit_shapes.items():
        hit_shapes_count_M[(BLOCK_X, BLOCK_Y, TILE_SIZE)] = {}
        hit_shapes_count_N[(BLOCK_X, BLOCK_Y, TILE_SIZE)] = {}
        for shape in shapes:
            M, N = shape
            if M not in hit_shapes_count_M[(BLOCK_X, BLOCK_Y, TILE_SIZE)]:
                hit_shapes_count_M[(BLOCK_X, BLOCK_Y, TILE_SIZE)][M] = 1
            else:
                hit_shapes_count_M[(BLOCK_X, BLOCK_Y, TILE_SIZE)][M] += 1
            if N not in hit_shapes_count_N[(BLOCK_X, BLOCK_Y, TILE_SIZE)]:
                hit_shapes_count_N[(BLOCK_X, BLOCK_Y, TILE_SIZE)][N] = 1
            else:
                hit_shapes_count_N[(BLOCK_X, BLOCK_Y, TILE_SIZE)][N] += 1
    np.save(file + "_hit_shapes_count_N.npy", hit_shapes_count_N)
    np.save(file + "_hit_shapes_count_M.npy", hit_shapes_count_M)

    # hit_shapes_range = sorted(hit_shapes_range.items(), key=lambda x: hit_times[x[0]], reverse=True)
    # hit_shapes_count_M = sorted(hit_shapes_count_M.items(), key=lambda x: hit_times[x[0]], reverse=True)
    # hit_shapes_count_N = sorted(hit_shapes_count_N.items(), key=lambda x: hit_times[x[0]], reverse=True)
    hit_shapes_range = sorted(hit_shapes_range.items(), key=lambda x: x[0])
    hit_shapes_count_M = sorted(hit_shapes_count_M.items(), key=lambda x: x[0])
    hit_shapes_count_N = sorted(hit_shapes_count_N.items(), key=lambda x: x[0])

    print("Results for ", file, ":")

    for BLOCK_X, count in hit_times_count_block_x:
        print(f"BLOCK_X={BLOCK_X}: Count={count}")
    print("")
    for BLOCK_Y, count in hit_times_count_block_y:
        print(f"BLOCK_Y={BLOCK_Y}: Count={count}")
    print("")
    for TILE_SIZE, count in hit_times_count_tile:
        print(f"TILE_SIZE={TILE_SIZE}: Count={count}")
    print("")

    for (BLOCK_X, BLOCK_Y, TILE_SIZE), shapes in hit_shapes_range:
        print(f"BLOCK_X={BLOCK_X}, BLOCK_Y={BLOCK_Y}, TILE_SIZE={TILE_SIZE}:")
        print(f"Min shape: {shapes[0]}")
        print(f"Max shape: {shapes[1]}")
        print("")

    for (BLOCK_X, BLOCK_Y, TILE_SIZE), shapes in hit_shapes_count_M:
        print(f"BLOCK_X={BLOCK_X}, BLOCK_Y={BLOCK_Y}, TILE_SIZE={TILE_SIZE}:")
        for M, count in shapes.items():
            print(f"M={M}: Count={count}")
        print("")

    # print(file + " Hit shapes count N", hit_shapes_count_N) 
    for (BLOCK_X, BLOCK_Y, TILE_SIZE), shapes in hit_shapes_count_N:
        print(f"BLOCK_X={BLOCK_X}, BLOCK_Y={BLOCK_Y}, TILE_SIZE={TILE_SIZE}:")
        for N, count in shapes.items():
            print(f"N={N}: Count={count}")
        print("")   



def search_schedule1(file):
    results = np.load(file + '_results.npy', allow_pickle=True)
    # top_results = np.load(file + '_top_results.npy', allow_pickle=True)

    best_schedule = {} # a dictionary to store the best schedule for each matrix size
    best_latency = {} # a dictionary to store the best latency for each matrix size
    for m in range(512, 8192+1, 512): # iterate over all possible values of M
        for n in range(512, 8192+1, 512): # iterate over all possible values of N
            best_schedule[(m, n)] = top_results[(m, n)][:3] # get the best schedule for (m, n) from top_results
            best_latency[(m, n)] = top_results[(m, n)][3] # get the best latency for (m, n) from top_results

    X = [] # a list to store the matrix sizes
    y = [] # a list to store the best schedule for each matrix size
    for m in range(512, 8192+1, 512): # iterate over all possible values of M
        for n in range(512, 8192+1, 512): # iterate over all possible values of N
            X.append([m, n]) # add [m, n] to the list
            y.append(list(best_schedule[(m, n)])) # add the best schedule for (m, n) to the list
    X = np.array(X) # convert the list to a numpy array
    y = np.array(y) # convert the list to a numpy array

    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(random_state=0) # create a DecisionTreeClassifier object
    tree.fit(X, y) # fit the data and the labels to the algorithm
    labels = tree.predict(X) # get the predicted labels for each matrix size

    range0 = [] # a list to store the matrix sizes in range0
    range1 = [] # a list to store the matrix sizes in range1
    range2 = [] # a list to store the matrix sizes in range2
    scheduleA = tuple(labels[0]) # assume the first label is the best schedule for range0
    scheduleB = tuple(labels[1]) # assume the second label is the best schedule for range1
    scheduleC = tuple(labels[2]) # assume the third label is the best schedule for range2
    for i in range(len(X)): # iterate over all matrix sizes
        if tuple(labels[i]) == scheduleA: # if the label is scheduleA
            range0.append(X[i]) # add the matrix size to range0
        elif tuple(labels[i]) == scheduleB: # if the label is scheduleB
            range1.append(X[i]) # add the matrix size to range1
        else: # otherwise
            range2.append(X[i]) # add the matrix size to range2

    # convert the lists to numpy arrays for better display
    range0 = np.array(range0)
    range1 = np.array(range1)
    range2 = np.array(range2)

    # print the ranges and schedules
    print("range0:", range0)
    print("scheduleA:", scheduleA)
    print("range1:", range1)
    print("scheduleB:", scheduleB)
    print("range2:", range2)
    print("scheduleC:", scheduleC)

# import numpy as np

# 加载已保存的结果
# results = np.load('results.npy', allow_pickle=True).item()
# # top_results = np.load('top_results.npy', allow_pickle=True).item()

# # 定义range0、range1和range2
# range0 = []
# range1 = []
# range2 = []

# # 遍历矩阵大小的范围，将每个矩阵大小添加到相应的range中
# for m in range(512, 8193, 512):
#     for n in range(512, 8193, 512):
#         if (m, n) in top_results:
#             range_latency = top_results[(m, n)][1]
#             if range_latency < 0.95 * results[(m, n)][top_results[(m, n)][0]]:
#                 range0.append((m, n))
#             elif range_latency < 1.05 * results[(m, n)][top_results[(m, n)][0]]:
#                 range1.append((m, n))
#             else:
#                 range2.append((m, n))
#         else:
#             range2.append((m, n))

# # 定义scheduleA、scheduleB和scheduleC
# scheduleA = []
# scheduleB = []
# scheduleC = []

# # 针对每个range的最佳方案，将其添加到相应的schedule列表中
# for (m, n) in range0:
#     scheduleA.append(top_results[(m, n)][0])
# for (m, n) in range1:
#     scheduleB.append(top_results[(m, n)][0])
# for (m, n) in range2:
#     scheduleC.append(top_results[(m, n)][0])

# print("range0:", range0)
# print("range1:", range1)
# print("range2:", range2)
# print("scheduleA:", scheduleA)
# print("scheduleB:", scheduleB)
# print("scheduleC:", scheduleC)



# file1 = "s1"
# # profile_schedule1(file1)
# # save_top_results()
# # # profile_candidates(file1)
# # profile_hit_shapes2("reduction2")
# # profile_hit_shapes1("reduction1")
# # search_schedule("reduction1")
# search_schedule1("reduction2")


# 加载已保存的结果
# results = np.load('reduction2_results.npy', allow_pickle=True).item()

# # 将结果转换为字典格式
# results_dict = {tuple(k): v for k, v in results.items()}

# # 生成top_results字典
# top_results = {}

# for (m, n), latency_dict in results_dict.items():
#     sorted_schedules = sorted(latency_dict.items(), key=lambda x: x[1])
#     top_results[(m, n)] = [schedule for schedule, latency in sorted_schedules[:3]]

# # 保存生成的top_results字典
# np.save('reduction2_top_results.npy', top_results)
# # results[(M, N)][(BLOCK_X, TILE_SIZE)] = latency
# 我们针对一个变长的矩阵设计了9种处理方案：schedule0, schedule1, ..., schedule8, 矩阵大小为M*N，M和N的范围都是[512, 8192]，步长为512。
# 每种方案在每种形状的矩阵下对应的处理时间已经测试了，记录在results字典中，数据结构为results[(M, N)][(schedule)] = latency，存成了numpy文件。
# 我们希望根据M和N的大小，自动选择一个最优的schedule，使得处理时间最短。
# 最终达到的效果是，保留3个左右的schedule，然后，你确定下来这样的逻辑：
# if ([m, n] in range0) {
#   schedule = scheduleA;
# } else if ([m, n] in range1) {
#   schedule = scheduleB;
# } else {
#   schedule = scheduleC;
# }
# 帮我求出range0, range1, range2, scheduleA, scheduleB, scheduleC，用python写就行。
# 对了，我还保存了一些中间结果，你可以直接用，比如每个矩阵对应的最优schedule和相应的latency，保存在top_results.npy中，格式是top_results[(M, N)] = [(schedule, latency)]

# import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier

# 加载已保存的结果
# results = np.load('reduction2_results.npy', allow_pickle=True).item()

# X = [] # a list to store the matrix sizes
# for m in range(512, 8192+1, 512): # iterate over all possible values of M
#   for n in range(512, 8192+1, 512): # iterate over all possible values of N
#     X.append([m, n]) # add [m, n] to the list
# X = np.array(X) # convert the list to a numpy array

# # results = np.load('results.npy', allow_pickle=True) # load the results dictionary
# best_schedule = {} # a dictionary to store the best schedule for each matrix size
# best_latency = {} # a dictionary to store the best latency for each matrix size
# for m in range(512, 8192+1, 512): # iterate over all possible values of M
#   for n in range(512, 8192+1, 512): # iterate over all possible values of N
#     best_schedule[(m, n)] = min(results[(m, n)], key=results[(m, n)].get) # get the best schedule for (m, n) from results
#     best_latency[(m, n)] = results[(m, n)][best_schedule[(m, n)]] # get the best latency for (m, n) from results

# y = [] # a list to store the best schedule for each matrix size
# for m in range(512, 8192+1, 512): # iterate over all possible values of M
#   for n in range(512, 8192+1, 512): # iterate over all possible values of N
#     y.append(best_schedule[(m, n)]) # add the best schedule for (m, n) to the list
# y = np.array(y) # convert the list to a numpy array

# tree = DecisionTreeClassifier(random_state=0) # create a DecisionTreeClassifier object
# tree.fit(X, y) # fit the data and the labels to the algorithm
# labels = tree.predict(X) # get the predicted labels for each matrix size

# range0 = [] # a list to store the matrix sizes in range0
# range1 = [] # a list to store the matrix sizes in range1
# range2 = [] # a list to store the matrix sizes in range2
# scheduleA = labels[0] # assume the first label is the best schedule for range0
# scheduleB = labels[1] # assume the second label is the best schedule for range1
# scheduleC = labels[2] # assume the third label is the best schedule for range2
# print(scheduleA, scheduleB, scheduleC)
# for i in range(len(X)): # iterate over all matrix sizes
# #   if labels[i] == scheduleA: # if the label is scheduleA
#   if np.array_equal(labels[i], scheduleA): 
#     range0.append(X[i]) # add the matrix size to range0
#   elif np.array_equal(labels[i], scheduleB): # if the label is scheduleB
#     range1.append(X[i]) # add the matrix size to range1
#   else: # otherwise
#     range2.append(X[i]) # add the matrix size to range2

# # convert the lists to numpy arrays for better display
# range0 = np.array(range0)
# range1 = np.array(range1)
# range2 = np.array(range2)

# # print the ranges and schedules
# print("range0:", range0)
# print("scheduleA:", scheduleA)
# print("range1:", range1)
# print("scheduleB:", scheduleB)
# print("range2:", range2)
# print("scheduleC:", scheduleC)


# scheduleA = (32, 8, 64)
# scheduleB = (32, 8, 128)
# scheduleC = (32, 16, 64)

# results = np.load('reduction2_results.npy', allow_pickle=True).item()
# # results[(m,n)][blocx,blocky,tilesize] = latency

# # (m1, n1, m2, n2)
# # ranges = [
# #     (512, 512, 1024, 1024),
# #     (1024, 1024, 2048, 2048),
# #     (2048, 2048, 2560, 2560),
# #     (2560, 2560,3072, 3072),
# #     (3072, 3072,3584, 3584),
# #     (3584, 3584,4096, 4096),
# #     (4096, 4096, 5120, 5120),
# #     (5120, 5120, 6144, 6144),
# #     (6144, 6144, 7168, 7168),
# #     (7168, 7168, 8192, 8192)
# # ]
# ranges = [
#     (512, 512, 1024, 1024),
#     (1024, 1024, 1024, 2048),
#     (1024, 2048, 2048, 2048),
#     (2048, 2048, 2048, 2560),
#     (2048, 2560, 2560, 2560),
#     (2560, 2560, 2560, 3072),
#     (2560, 3072, 3072, 3072),
#     (3072, 3072, 3072, 3584),
#     (3072, 3584, 3584, 3584),
#     (3584, 3584, 3584, 4096),
#     (3584, 4096, 4096, 4096),
#     (4096, 4096, 4096, 5120),
#     (4096, 5120, 5120, 5120),
#     (5120, 5120, 5120, 6144),
#     (5120, 6144, 6144, 6144),
#     (6144, 6144, 6144, 7168),
#     (6144, 7168, 7168, 7168),
#     (7168, 7168, 7168, 8192),
#     (7168, 8192, 8192, 8192)
# ]

schedules2 = [
    (32, 8, 64),
    (32, 8, 128),
    (32, 16, 64),
    (16, 8,  64),
    (16, 8,  128),
    (16, 16, 64),
    (16, 16, 128),
    (32, 8, 256),
    (32, 16, 128),
    (32, 16, 256),
    (32, 32, 64),
    (32, 32, 128),
    (32, 32, 256),
    (16, 8, 256)
]


# #  for each range, find the best 3 schedules that has the minimum average latency  of the range
# best_schedules = {}
# for (m1, n1, m2, n2) in ranges:
#     best_schedules[(m1, n1, m2, n2)] = []
#     for schedule in schedules:
#         latencies = 0.0
#         if schedule[1] * schedule[2] > m1:
#             continue
#         for m in range(m1, m2+1, 512):
#             for n in range(n1, n2+1, 512):
#                 latencies += results[(m, n)][schedule]
#         best_schedules[(m1, n1, m2, n2)].append((schedule, latencies))
#     best_schedules[(m1, n1, m2, n2)] = sorted(best_schedules[(m1, n1, m2, n2)], key=lambda x: x[1])[:3]

# # print the best schedules for each range
# for (m1, n1, m2, n2), schedules in best_schedules.items():
#     print(f"Range: ({m1}, {n1}, {m2}, {n2})")
#     for schedule, latency in schedules:
#         print(f"Schedule: {schedule}, Latency: {latency}")
#     print("")

# if M <= 2560 and N <= 2560:
#     block_x, block_y, tilesize = (16, 8, 64)
# elif M <= 8192 and N <= 8192:
#     block_x, block_y, tilesize = (32, 8, 64)


results1 = np.load('reduction1_results.npy', allow_pickle=True).item()
results2 = np.load('reduction2_results.npy', allow_pickle=True).item()
# ranges = [
#     (512, 512, 1024, 1024),
#     (1024, 1024, 2048, 2048),
#     (2048, 2048, 2560, 2560),
#     (2560, 2560,3072, 3072),
#     (3072, 3072,3584, 3584),
#     (3584, 3584,4096, 4096),
#     (4096, 4096, 5120, 5120),
#     (5120, 5120, 6144, 6144),
#     (6144, 6144, 7168, 7168),
#     (7168, 7168, 8192, 8192)
# ]
ranges = [
    (512, 512, 1024, 1024),
    (1024, 1024, 1024, 2048),
    (1024, 2048, 2048, 2048),
    (2048, 2048, 2048, 2560),
    (2048, 2560, 2560, 2560),
    (2560, 2560, 2560, 3072),
    (2560, 3072, 3072, 3072),
    (3072, 3072, 3072, 3584),
    (3072, 3584, 3584, 3584),
    (3584, 3584, 3584, 4096),
    (3584, 4096, 4096, 4096),
    (4096, 4096, 4096, 5120),
    (4096, 5120, 5120, 5120),
    (5120, 5120, 5120, 6144),
    (5120, 6144, 6144, 6144),
    (6144, 6144, 6144, 7168),
    (6144, 7168, 7168, 7168),
    (7168, 7168, 7168, 8192),
    (7168, 8192, 8192, 8192)
]

schedules1 = [
    (32, 32),
    (64, 32),
    (128, 32),
    (256, 32),
    (512, 32),
    (64, 64),
    (128, 64),
    (256, 64),
    (512, 64),
]


#  for each range, find the best 3 schedules that has the minimum average latency  of the range
best_schedules = {}
for (m1, n1, m2, n2) in ranges:
    best_schedules[(m1, n1, m2, n2)] = []
    for schedule in schedules1:
        latencies = 0.0
        if schedule[1] > m1:
            continue
        for m in range(m1, m2+1, 512):
            for n in range(n1, n2+1, 512):
                latencies += results1[(m, n)][schedule]
        best_schedules[(m1, n1, m2, n2)].append((schedule, latencies / ((m2 - m1 + 512) * (n2 - n1 + 512) / (512 * 512))))
    for schedule in schedules2:
        latencies = 0.0
        if schedule[1] * schedule[2] > m1:
            continue
        for m in range(m1, m2+1, 512):
            for n in range(n1, n2+1, 512):
                latencies += results2[(m, n)][schedule]
        best_schedules[(m1, n1, m2, n2)].append((schedule, latencies / ((m2 - m1 + 512) * (n2 - n1 + 512) / (512 * 512))))
    best_schedules[(m1, n1, m2, n2)] = sorted(best_schedules[(m1, n1, m2, n2)], key=lambda x: x[1])[:3]

# print the best schedules for each range
for (m1, n1, m2, n2), schedules in best_schedules.items():
    print(f"Range: ({m1}, {n1}, {m2}, {n2})")
    for schedule, latency in schedules:
        print(f"Schedule: {schedule}, Latency: {latency}")
    print("")

# if M <= 2560 and N <= 2560:
#     block_x, tilesize = (256, 32)
# elif M <= 4096 and N <= 5120:
#     block_x, tilesize = (512, 32)
# elif M <= 8192 and N <= 8192:
#     block_x, tilesize = (512, 64)

# overall results 
# if M <= 2560 and N <= 2560:
#     schedule = schedule1(block_x = 256, tilesize = 32)
# elif M <= 4096 and N <= 4096:
#     schedule = schedule1(block_x = 512, tilesize = 32)
# elif M <= 8192 and N <= 8192:
#     schedule = schedule2(block_x = 32, block_y = 8, tilesize = 64)