

class TestResult:
    algorithm_name: str
    number_of_cities: int
    time_taken: float


class Simulation:
    def __init__(self):
        self.results = []
        
    def save_test_result(self, algorithm_name: str, number_of_cities: int, time_taken: float):
        results = TestResult()
        
        results.algorithm_name = algorithm_name
        results.number_of_cities = number_of_cities
        results.time_taken = time_taken
        
        self.results.append(results)
        
        return self.results[len(self.results)-1]