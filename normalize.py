import math

def get_variance(Features, avg):

    total = 0
    count = 0

    for feature in Features:
        for num in feature:
            num -= avg
            num = num * num
            total += num
            count += 1

    variance = total / count
    return variance

def get_average(Features):

    total = 0
    count = 0

    for feature in Features:
        total += sum(feature)
        count += 7

    return total / count


def get_standard_deviation(Features, avg):

    total = 0
    count = 0

    for feature in Features:
        for num in feature:
            num -= avg
            num = num * num
            total += num
            count += 1

    variance = total / count
    std_dev = math.sqrt(variance)

    return std_dev


def get_standard_distribution(Features, avg, std_dev):

    new = []

    for feature in Features:
        new_feature = []
        for num in feature:
            num -= avg
            num = num / std_dev
            new_feature.append(num)
        new.append(new_feature)

    return new