from dataset.kitti_dataset import SemKITTI
from dataset.carla_dataset import CarlaDataset

def dataset_builder(args):
    print("build dataset")
    if args.dataset == 'kitti':
        dataset = SemKITTI(args, 'train')
        val_dataset = SemKITTI(args, 'val')
        args.num_class = 20
        args.grid_size = [256, 256, 32]
        class_names = [
                'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'bicyclist',
                'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence',
                'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign'
            ]
    elif args.dataset == 'carla':
        dataset = CarlaDataset(args, 'train')
        val_dataset = CarlaDataset(args, 'val')
        args.num_class = 11 
        args.grid_size = [128, 128, 8]
        class_names = ['building', 'barrier', 'other', 'pedestrian', 'pole', 'road', 'ground', 'sidewalk', 'vegetation', 'vehicle']
        
    return dataset, val_dataset, args.num_class, class_names