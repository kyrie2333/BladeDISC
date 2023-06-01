import subprocess
import numpy as np


def get_time2(M, N):
    command = ["nsys", "profile", "--stats=true", "--output=profile_output", "./colreduct_all", str(M), str(N)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    output = output.decode("utf-8")
    if output == "":
        return []
    
    lines = output.split("\n")
    for i in range(len(lines)):
        if lines[i].startswith("[6/8]"):
            break
    for j in range(len(lines)):
        if lines[j].startswith("[7/8]"):
            break
    lines = lines[i+4:j-1]
    # print(lines)
    ranks = {}
    for line in lines:
        # get colreduct2_32x8x64
        schedule = line.split("(")[0].strip()
        # check boundary
        # if schedule.split()[8] exists
        #
        if len(schedule.split()) >= 9:
            schedule = schedule.split()[8]
        # print(schedule)
        if len(line.split()) >= 4:
            time = float(line.split()[3]) / 1000
        ranks[schedule] = (time, 0.0)

    ranks = sorted(ranks.items(), key=lambda x: x[1])
    # find the index of colreduct_disc
    for i in range(len(ranks)):
        if ranks[i][0] == "colreduct_disc":
            break
    
    disc_time = ranks[i][1][0]
    
    for j in range(len(ranks)):
        # ranks[j][1][1] = ranks[j][1][0] / ranks[i][1][0]
        time = ranks[j][1][0]
        ranks[j] = (ranks[j][0], (time / disc_time)) # the smaller the better
        # print(ranks[j][1])

    return ranks

def get_results():
    results = {}
    for M in range(512, 8192 + 1, 512):
        for N in range(512, 8192 + 1, 512):
            results[(M, N)] = get_time2(M, N)
            print("Results for (M, N) =", (M, N))
            for rank in results[(M, N)]:
                print(rank)
            print()

    for M in range(8192, 196608 + 1, 2048):
        for N in range(8192, 196608 + 1, 2048):
            results[(M, N)] = get_time2(M, N)
            print("Results for (M, N) =", (M, N))
            for rank in results[(M, N)]:
                print(rank)
            print()
    
    np.save("reduction_results_compare.npy", results)




# Results for (M, N) = (8192, 65536)
# ('colreduct1_256x32', 0.2960855705707746)
# ('colreduct1_512x32', 0.29662914695393383)
# ('colreduct2_32x8x64', 0.30791223959293895)
# ('colreduct_disc', 1.0)

# Results for (M, N) = (8192, 67584)
# ('colreduct1_256x32', 0.29611898886221744)
# ('colreduct1_512x32', 0.296268665093819)
# ('colreduct2_32x8x64', 0.30299637783519523)
# ('colreduct_disc', 1.0)
# read from log file 
def get_results_from_log():
    lines = open("compare_schedule_disc.log").readlines()
    results = {}
    for line in lines:
        if line.startswith("Results"):
            M = int(line.split("=")[1].split(",")[0].split("(")[1].strip())
            N = int(line.split("=")[1].split(",")[1].split(")")[0].strip())
            results[(M, N)] = []
        elif line.startswith("("):
            schedule = line.split(",")[0].strip("()").strip("'")
            time = float(line.split(",")[1].split(")")[0].strip())
            # print(schedule, time)
            results[(M, N)].append((schedule, time))
    
    return results



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
    (7168, 7168, 8192, 8192),
    (8192, 8192, 9216, 10240),
    (9216, 10240, 11264, 11264),
    (11264, 11264, 12288, 12288),
    (12288, 12288, 12288, 17480)

]
# 统计每个range内每个schedule的命中次数
results = get_results_from_log()

hit_times = {}
for (m1, n1, m2, n2) in ranges:
    hit_times[(m1, n1, m2, n2)] = {}
    for M in range(m1, m2 + 1, 512):
        for N in range(n1, n2 + 1, 512):
            if (M, N) not in results:
                continue
            schedule1 = results[(M, N)][0][0]
            schedule2 = results[(M, N)][1][0]
            # print(schedule)
            if schedule1 not in hit_times[(m1, n1, m2, n2)]:
                hit_times[(m1, n1, m2, n2)][schedule1] = 1
            else:
                hit_times[(m1, n1, m2, n2)][schedule1] += 1
            if schedule2 not in hit_times[(m1, n1, m2, n2)]:
                hit_times[(m1, n1, m2, n2)][schedule2] = 1
            else:
                hit_times[(m1, n1, m2, n2)][schedule2] += 1

# print(hit_times)
for (m1, n1, m2, n2) in ranges:
    print("Range:", (m1, n1, m2, n2))
    for schedule, times in sorted(hit_times[(m1, n1, m2, n2)].items(), key=lambda x: x[1], reverse=True):
        print(schedule, times)
    print()
