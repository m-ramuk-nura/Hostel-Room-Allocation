from clustering import StudentClusterGenerator
from health_ranker import HealthConditionRanker
from room_allocation import RoomAllocator

def main():
 
    cluster_obj = StudentClusterGenerator(
        input_path='Dataset/final.csv',
        output_dir='clusters',
        optimal_k=20
    )

    cluster_obj.run()

    ranker = HealthConditionRanker(
        dataset_path='Dataset/final.csv',
        folder_path='clusters'
    )
    ranker.run()

    allocator = RoomAllocator(
        folder_path='clusters',           
        ranking_col='Health_Condition_Rank', 
        room_col='room_number',           
        room_size=4                       
    )

    allocator.run()

 

if __name__ == "__main__":
    main()
