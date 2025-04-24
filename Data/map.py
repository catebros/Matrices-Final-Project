import folium
import io
import pandas as pd
import os

class GreatBritainMap():
    def __init__(self, y_coord = -2.2360, x_coord = 54.0781):
        if y_coord is None or x_coord is None:
            raise ValueError("You need to add y and x coordinates of the location for the map")

        self._center = (x_coord, y_coord)  # <-- FIX: this was missing
        self.map = folium.Map(
            location=[x_coord, y_coord],
            zoom_start=6,
            dragging=False,
            scrollWheelZoom=False,
            doubleClickZoom=False,
            touchZoom=False,
            zoom_control=False
        )
        
        self.current_dir = os.path.dirname(__file__)
        csv_path = os.path.join(self.current_dir, "UK_Cities.csv")
        self.all_cities = pd.read_csv(csv_path)  # store full list here
        self.cities = self.all_cities.copy()

        self.load_and_plot_cities()

    def load_and_plot_cities(self):
        try:
            for _, row in self.cities.iterrows():
                folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    popup=row['City'],
                    icon=folium.Icon(color='red')
                ).add_to(self.map)
        except Exception as e:
            print(f"Failed to load or plot cities: {e}")

    def display_map(self):
        map_bytes = io.BytesIO()
        self.map.save(map_bytes, close_file=False)
        return map_bytes.getvalue().decode('utf-8')

    def draw_tsp_path(self, ordered_cities):
        """Draw lines connecting cities in the TSP order."""
        path = [
            (row['Latitude'], row['Longitude']) 
            for _, row in ordered_cities.iterrows()
        ]
        folium.PolyLine(path, color="blue", weight=3, opacity=0.7).add_to(self.map)

    def get_dataset(self) -> pd.DataFrame:
        return self.cities
    
    def generate_uk_map(self, num_cities: int):
        """Sample num_cities randomly from the UK dataset and redraw the map."""
        if num_cities < 1 or num_cities > len(self.all_cities):
            raise ValueError(f"num_cities must be between 1 and {len(self.all_cities)}")

        # Sample cities and assign to current working set
        self.cities = self.all_cities.sample(n=num_cities, random_state=None).reset_index(drop=True)

        # Recreate map and plot only sampled cities
        self.map = folium.Map(
            location=[self._center[0], self._center[1]],
            zoom_start=6,
            dragging=False,
            scrollWheelZoom=False,
            doubleClickZoom=False,
            touchZoom=False,
            zoom_control=False
        )

        self.load_and_plot_cities()