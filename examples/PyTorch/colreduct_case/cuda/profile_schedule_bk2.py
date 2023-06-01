import subprocess
import numpy as np


def get_time2(M, N, BLOCK_X, BLOCK_Y, TILE_SIZE, filename):
    # subprocess.run(["nvcc", filename+".cu", "-o", filename])

    command = ["./"+filename, str(M), str(N), str(BLOCK_X), str(BLOCK_Y), str(TILE_SIZE)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    output = output.decode("utf-8")
    # error = error.decode("utf-8")
    print(M, N, BLOCK_X, BLOCK_Y, TILE_SIZE, output)

    latency = 0.0
    if len(output.strip().split("\n")[0].split()) > 1:
        latency = float(output.strip().split("\n")[0].split()[1])
    # is_match = "NotMatch"
    # if "Wrong" in output:
        # latency = 0.0
    return latency


def profile_schedule2(file):
    # results = {}
    # blocksize = [(8, 16), (8, 32),  
    #              (16, 8), (16, 16), (16, 32),  
    #              (32, 8), (32, 16), (32, 32)
    #             ]
    # for M in range(512, 8192 + 1, 512):
    #     for N in range(512, 8192 + 1, 512):
    #         results[(M, N)] = {}  
    #         for (BLOCK_X, BLOCK_Y) in blocksize:
    #             for TILE_SIZE in range(64 , 8192 + 1, 64):
    #                 if BLOCK_Y * TILE_SIZE > M:
    #                     continue
    #                 latency = get_time2(M, N, BLOCK_X, BLOCK_Y, TILE_SIZE, file)
    #                 if float(latency) > 0.00001:
    #                     results[(M, N)][(BLOCK_X, BLOCK_Y, TILE_SIZE)] = latency

    # # print("Results", results)
    # np.save("reduction2_results.npy", results)
    results = np.load("reduction2_results.npy", allow_pickle=True).item()

    candidates = {}
    top_result = {}
    for (M, N), block_results in results.items():
        sorted_results = sorted(block_results.items(), key=lambda x: x[1])
        top_result[(M,N)] = sorted_results[0]
        print("Top result for (M, N) =", top_result)
        for (BLOCK_X, BLOCK_Y, TILE_SIZE), latency in sorted_results:
            if len(top_result) > 1 and float(latency) <= float(top_result[1]) / 0.95:
                if (BLOCK_X, BLOCK_Y, TILE_SIZE) not in candidates:
                    candidates[(BLOCK_X, BLOCK_Y, TILE_SIZE)] = 1
                else:
                    candidates[(BLOCK_X, BLOCK_Y, TILE_SIZE)] += 1
    np.save("reduction2_candidates.npy", candidates)
    np.save("reduction2_top_results.npy", top_result)
    best_candidate = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    print("Best candidates for schedule 2:")
    for (BLOCK_X, BLOCK_Y, TILE_SIZE), count in best_candidate:
        print(f"BLOCK_X={BLOCK_X}, BLOCK_Y={BLOCK_Y}, TILE_SIZE={TILE_SIZE}: Count={count}")
    
    np.save("reduction2_best_candidates.npy", best_candidate)
    

def profile_candidates(file):
    results = {}
    candidates = [
        (32, 8, 64),
        (16, 8, 64),
        (32, 8, 128),
        (32, 16, 64),
        (32, 8, 192),
        (32, 8, 256),
        (32, 16, 128),
        (16, 16, 64),
        (32, 8, 320)
    ]
    # total_results = np.load("reduction2_results.npy", allow_pickle=True).item()
    # for N in range(512, 8192 + 1, 512):
    #     for M in range(512, 8192 + 1, 512):
    #         results[(M, N)] = {}  
    #         for (BLOCK_X, BLOCK_Y, TILE_SIZE) in candidates:
    #             if BLOCK_Y * TILE_SIZE > M:
    #                 results[(M, N)][(BLOCK_X, BLOCK_Y, TILE_SIZE)] = total_results[(M, N)][(BLOCK_X, BLOCK_Y, TILE_SIZE)]

    # np.save("reduction2_results2.npy", results)
    results = np.load("reduction2_results.npy", allow_pickle=True).item()

    hit_times = {}
    hit_shapes = {}
    top_results = np.load("reduction2_top_results.npy", allow_pickle=True).item()
    for (M, N), block_results in results.items():
        # if block_results not in candidates:
            
        sorted_results = sorted(block_results.items(), key=lambda x: x[1])
        if len(sorted_results[0]) < 2:
            continue
        # top_result = top_results[(M,N)][0][3]
        top_result = float(sorted_results[0][1])
        # if top_result < float(sorted_results[0][1]):
        #     top_result = float(sorted_results[0][1]) / 0.9
        # print("Top result for (M, N) =", (M, N), ":", top_results)
        for (BLOCK_X, BLOCK_Y, TILE_SIZE), latency in sorted_results:
            if latency < top_result / 0.9:
                # continue
                if (BLOCK_X, BLOCK_Y, TILE_SIZE) not in hit_times:
                    hit_times[(BLOCK_X, BLOCK_Y, TILE_SIZE)] = 1
                    hit_shapes[(BLOCK_X, BLOCK_Y, TILE_SIZE)] = [(M, N)]
                else:
                    hit_times[(BLOCK_X, BLOCK_Y, TILE_SIZE)] += 1
                    hit_shapes[(BLOCK_X, BLOCK_Y, TILE_SIZE)].append((M, N))
        # hit_shapes = sorted(hit_shapes.items(), key=lambda x: x[1], reverse=True)
    
    np.save("reduction2_hit_times.npy", hit_times)
    np.save("reduction2_hit_shapes.npy", hit_shapes)
    hit_times = sorted(hit_times.items(), key=lambda x: x[1], reverse=True)
    print("Hit times", hit_times)
    print("Hit shapes", hit_shapes)
    # best_candidate = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    # print("Best candidates for schedule 2:")
    for (BLOCK_X, BLOCK_Y, TILE_SIZE), count in hit_times:
        print(f"BLOCK_X={BLOCK_X}, BLOCK_Y={BLOCK_Y}, TILE_SIZE={TILE_SIZE}: Count={count}")
    
# read top results from txt file 
def save_top_results():
    f = open("profile_reduct2_res.txt", "r")
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
            # top_results[(M, N)] = {}
            # delete ( and )
            part = part.strip().strip("()").strip(",")
            # print(part)
            idx, time = part.split(")")
            # print(key, value)
            # key = tuple(int(x) for x in key.split(","))
            blockx, blocky, tilesize = idx.split(",")
            blockx = int(blockx)
            blocky = int(blocky)
            tilesize = int(tilesize)
            time = time.strip(",")
            time = float(time)
            if idx not in top_results:
                top_results[(M, N)] = {}
                top_results[(M, N)] = [(blockx, blocky, tilesize, time)]

    f.close()
    np.save("reduction2_top_results.npy", top_results)
    # print(top_results)
    # get time of top_results[(512, 512)]
    # print(top_results[(512, 512)][0][3])



# file1 = "s1"
# profile_schedule1(file1)
file2 = "s2"
# save_top_results()
# profile_schedule2(file2)
profile_candidates(file2)
# results = {
#     (512, 512): {(16, 8, 64): 4.6592, (32, 8, 64): 5.40672},
#     (512, 1024): {(16, 8, 64): 8.192, (32, 8, 64): 9.216}
# }
# # save the results to a file
# np.save("reduction2_results.npy", results)
# # load the results from a file
# results = np.load("reduction2_results.npy", allow_pickle=True).item()


