class SuperGraph:
    def __init__(self, num_vertices):
        self.N = num_vertices
        self.structure = []

    def super_addition(self, alpha, beta, gamma):
        self.structure.append([alpha, beta, gamma])

    def display_result(self, distance_list):
        print("Node Distance from Origin:")
        for i in range(self.N):
            print(f"{i}\t\t{distance_list[i]}")

    def unveil_negative_sequence(self, precursor, initial):
        print("Detected detrimental loop:")
        sequence = []
        sequence.append(initial)
        current_step = precursor[initial]
        while current_step != initial:
            sequence.append(current_step)
            current_step = precursor[current_step]
        sequence.append(initial)
        sequence.reverse()
        print(" -> ".join(map(str, sequence)))

    def magic_journey(self, origin):
        distance_list = [float("Inf")] * self.N
        precursor = [-1] * self.N
        distance_list[origin] = 0

        for _ in range(self.N - 1):
            for alpha, beta, gamma in self.structure:
                if distance_list[alpha] != float("Inf") and distance_list[alpha] + gamma < distance_list[beta]:
                    distance_list[beta] = distance_list[alpha] + gamma
                    precursor[beta] = alpha

        # Check for detrimental loops
        for alpha, beta, gamma in self.structure:
            if distance_list[alpha] != float("Inf") and distance_list[alpha] + gamma < distance_list[beta]:
                print("Graph contains detrimental loop")
                self.unveil_negative_sequence(precursor, beta)
                return

        self.display_result(distance_list)

# Example usage:
s = SuperGraph(5)
s.super_addition(0, 1, -1)
s.super_addition(0, 2, 4)
s.super_addition(1, 2, 3)
s.super_addition(1, 3, 2)
s.super_addition(1, 4, 2)
s.super_addition(3, 2, 5)
s.super_addition(3, 1, 1)
s.super_addition(4, 3, -3)

s.magic_journey(0)