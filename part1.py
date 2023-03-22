import torch
import matplotlib.pyplot as plt


def main():
    words = open("names.txt").read().splitlines()
    print(words[:10])

    len_min = min(len(w) for w in words)
    len_max = max(len(w) for w in words)

    print("Min length:", len_min)
    print("Max length:", len_max)



    N = torch.zeros(27, 27, dtype=torch.int32)

    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            N[ix1, ix2] += 1

    # plt.figure(figsize=(16,16))
    # plt.imshow(N, cmap="Blues")
    # for i in range(27):
    #     for j in range(27):
    #         chstr = itos[i] + itos[j]
    #         plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
    #         plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")
    # plt.axis('off')
    #
    # plt.imshow(N)
    # plt.show()

    g = torch.Generator().manual_seed(2147483647)
    out = []

    P = (N+1).float()
    P /= P.sum(1, keepdim=True)

    log_likelihood = 0.0
    n = 0

    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            prob = P[ix1, ix2]
            logprob = torch.log(prob)
            log_likelihood += logprob
            n += 1


    print(f"{log_likelihood=}")
    nll = -log_likelihood
    print(f"{nll=}")
    print(f"{nll/n}")

    for _ in range(20):
        i = 0
        while True:
            p = P[i]
            i = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(itos[i])
            if i == 0:
                break

    print(''.join(out))

if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
