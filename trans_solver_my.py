import random
from concurrent.futures import ProcessPoolExecutor, as_completed

def find_error(E7, I7, K7, L7, O7, O8, P6, P7, R6, R7, R8):
    #E6=P6+P7-E7
    E8=O7+O8-E7
    I6=P6+P7-I7
    I8=O7+O8-I7
    K8=O7+O8-K7
    L6=P6+P7-L7

    err=0
    err+=abs(3.69 - (O8/O7)*(R7/R8))
    err+=abs(2.15 - (E8/E7)*(R7/R8))
    err+=abs(1.44 - (K8/K7)*(R7/R8))
    err+=abs(1.06 - (I8/I7)*(R7/R8))
    err+=abs(0.8  - (P6/P7)*(R7/R6))
    err+=abs(0.63 - (I6/I7)*(R7/R6))
    err+=abs(0.51 - (L6/L7)*(R7/R6))

    return err


E7, I7, K7, L7, O7, O8, P6, P7, R6, R7, R8 = [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]

def _anneal_run(seed, steps, domain):
    random.seed(seed)

    def clamp(v):
        return min(max(v, domain.start), domain.stop - 1)

    def random_solution():
        return [random.choice(domain) for _ in range(11)]

    current = random_solution()
    current_err = find_error(*current)
    best_err, best_solution = current_err, tuple(current)

    temp = 1.0
    cooling = 0.999

    for i in range(steps):
        neighbor = current[:]
        idx = random.randrange(len(neighbor))
        neighbor[idx] = clamp(neighbor[idx] + random.choice([-1, 1]))
        err = find_error(*neighbor)

        if err < current_err or random.random() < pow(2.71828, -(err - current_err) / temp):
            current, current_err = neighbor, err
            if err < best_err:
                best_err, best_solution = err, tuple(neighbor)

        temp *= cooling

    return best_solution, best_err


def find():
    domain = range(15, 69)

    best_err = float("inf")
    best_solution = None

    runs = 3000
    steps = 10000

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(_anneal_run, random.randrange(1_000_000_000), steps, domain) for _ in range(runs)]
        for future in as_completed(futures):
            solution, err = future.result()
            if err < best_err:
                best_err, best_solution = err, solution
                print("best err:", err)

    return best_solution, best_err

if __name__ == "__main__":
    best, err = find()
    print("Best solution:", best)
    print("Error:", err)

print("Final values:", end="\0")
#print()
print("Final values:", end="\0")