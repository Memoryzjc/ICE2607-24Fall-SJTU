import os
import numpy as np
import matplotlib.pyplot as plt
from extract_feature_resnet50 import extract_features_resnet50
from extract_feature_vgg16 import extract_features_vgg16
from extract_feature_inception import extract_features_inception

# Calculate the similarity between two features using angle
def cal_similarity(feature1, feature2):
    feature1 = feature1.flatten()
    feature2 = feature2.flatten()
    similarity = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
    return np.arccos(similarity) / np.pi * 180


# Extract features of all images in the dataset using ResNet50
def extract_dataset_feature_resnet50(dataset_path, save_path):
    for i in range(1, 51):
        input_image_path = dataset_path + f'/{i}.jpg'
        save_path_i = save_path + f'/{i}.npy'
        extract_features_resnet50(input_image_path, save_path_i)


# Extract features of all images in the dataset using VGG16
def extract_dataset_feature_vgg16(dataset_path, save_path):
    for i in range(1, 51):
        input_image_path = dataset_path + f'/{i}.jpg'
        save_path_i = save_path + f'/{i}.npy'
        extract_features_vgg16(input_image_path, save_path_i)


def extract_dataset_feature_inception(dataset_path, save_path):
    for i in range(1, 51):
        input_image_path = dataset_path + f'/{i}.jpg'
        save_path_i = save_path + f'/{i}.npy'
        extract_features_inception(input_image_path, save_path_i)


# Query the most similar image in the dataset
def query_image(query_image_feature_path, dataset_path):
    query_feature = np.load(query_image_feature_path)

    min_similarity = 180
    min_index = -1
    for i in range(1, 51):
        feature_path = dataset_path + f'/{i}.npy'
        feature = np.load(feature_path)
        if feature is None:
            print('Feature not found!')
            continue

        sim = cal_similarity(query_feature, feature)
        if sim < min_similarity:
            min_similarity = sim
            min_index = i
    return min_index, min_similarity

# Query the k most similar images in the dataset
def query_k_image(query_image_feature_path, dataset_path, k=5):
    query_feature = np.load(query_image_feature_path)

    similarity = []
    index = []

    for i in range(1, 51):
        feature_path = dataset_path + f'/{i}.npy'
        feature = np.load(feature_path)
        if feature is None:
            print('Feature not found!')
            continue

        sim = cal_similarity(query_feature, feature)
        similarity.append(sim)
        index.append(i)
    
    similarity = np.array(similarity)
    index = np.array(index)

    min_index = index[np.argsort(similarity)[:k]]
    min_similarity = similarity[np.argsort(similarity)[:k]]

    return min_index, min_similarity

def main():
    os.makedirs('./Feature_resnet50', exist_ok=True)
    os.makedirs('./Feature_vgg16', exist_ok=True)
    os.makedirs('./Feature_inception', exist_ok=True)
    os.makedirs('./Query_feature', exist_ok=True)
    os.makedirs('./Output', exist_ok=True)

    dataset_path = './Dataset'
    save_path_resnet50 = './Feature_resnet50'
    save_path_vgg16 = './Feature_vgg16'
    save_path_inception = './Feature_inception'
    output_path = './Output'

    # If the features of the dataset have been extracted, comment the following three lines
    extract_dataset_feature_resnet50(dataset_path, save_path_resnet50)
    extract_dataset_feature_vgg16(dataset_path, save_path_vgg16)
    extract_dataset_feature_inception(dataset_path, save_path_inception)

    query_images = ['./1.jpg', './2.jpg', './3.jpg']
    for i, query_image_path in enumerate(query_images):
        query_image_path = f'./Query_image/{i+1}.jpg'

        print('\n')
        print('Query the most similar image using resnet50!')

        # Extract features of the query image using resnet50
        extract_features_resnet50(query_image_path, f'./Query_feature/query_resnet50_{i}.npy')

        # Query the most similar image using resnet50
        query_image_feature_path = f'./Query_feature/query_resnet50_{i}.npy'
        min_k_index_resnet50, min_k_similarity_resnet50 = query_k_image(query_image_feature_path, './Feature_resnet50', k=5)

        print('\n')
        print('Query the most similar image using vgg16!')

        # Extract features of the query image using vgg16
        extract_features_vgg16(query_image_path, f'./Query_feature/query_vgg16_{i}.npy')

        # Query the most similar image using vgg16
        query_image_feature_path = f'./Query_feature/query_vgg16_{i}.npy'
        min_k_index_vgg16, min_k_similarity_vgg16 = query_k_image(query_image_feature_path, './Feature_vgg16', k=5)

        print('\n')
        print('Query the most similar image using inception!')

        # Extract features of the query image using inception
        extract_features_inception(query_image_path, f'./Query_feature/query_inception_{i}.npy')

        # Query the most similar image using inception
        query_image_feature_path = f'./Query_feature/query_inception_{i}.npy'
        min_k_index_inception, min_k_similarity_inception = query_k_image(query_image_feature_path, './Feature_inception', k=5)

        print('\n')

        print('The query image is:', query_image_path)
        print('The top 5 most similar images using ResNet50:')

        plt.figure(figsize=(28, 5))
        plt.subplot(1, 6, 1)
        query_image = plt.imread(query_image_path)
        plt.imshow(query_image)
        plt.axis('off')
        plt.title('Query Image')
        for j, (idx, sim) in enumerate(zip(min_k_index_resnet50, min_k_similarity_resnet50)):
            print('Index:', idx, 'Similarity:', sim)
            image_path = f'./Dataset/{idx}.jpg'
            image = plt.imread(image_path)
            plt.subplot(1, 6, j+2)
            plt.imshow(image)
            plt.axis('off')
            plt.title(f'Top {j+1} similar: Index: {idx}, Similarity: {sim:.2f}')
        plt.axis('off')
        plt.suptitle('ResNet50')
        plt.tight_layout()
        plt.savefig(f'{output_path}/ResNet50_{i}.png')
        
        print('The top 5 most similar images using VGG16:')

        plt.figure(figsize=(28, 5))
        plt.subplot(1, 6, 1)
        query_image = plt.imread(query_image_path)
        plt.imshow(query_image)
        plt.axis('off')
        plt.title('Query Image')
        for j, (idx, sim) in enumerate(zip(min_k_index_vgg16, min_k_similarity_vgg16)):
            print('Index:', idx, 'Similarity:', sim)
            image_path = f'./Dataset/{idx}.jpg'
            image = plt.imread(image_path)
            plt.subplot(1, 6, j+2)
            plt.imshow(image)
            plt.axis('off')
            plt.title(f'Top {j+1} similar: Index: {idx}, Similarity: {sim:.2f}')
        plt.axis('off')
        plt.suptitle('VGG16')
        plt.tight_layout()
        plt.savefig(f'{output_path}/VGG16_{i}.png')

        print('The top 5 most similar images using Inception V3:')
        
        plt.figure(figsize=(28, 5))
        plt.subplot(1, 6, 1)
        query_image = plt.imread(query_image_path)
        plt.imshow(query_image)
        plt.axis('off')
        plt.title('Query Image')
        for j, (idx, sim) in enumerate(zip(min_k_index_inception, min_k_similarity_inception)):
            print('Index:', idx, 'Similarity:', sim)
            plt.subplot(1, 6, j+2)
            image_path = f'./Dataset/{idx}.jpg'
            image = plt.imread(image_path)
            plt.imshow(image)
            plt.axis('off')
            plt.title(f'Top {j+1} similar: Index: {idx}, Similarity: {sim:.2f}')
        plt.axis('off')
        plt.suptitle('Inception V3')
        plt.tight_layout()
        plt.savefig(f'{output_path}/Inception_{i}.png')


if __name__ == '__main__':
    main()