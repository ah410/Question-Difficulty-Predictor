from data import load_student_data
from validation import k_fold_cross_validation
from train_and_predict import train_and_predict

if __name__ == '__main__':
    student_dataset = load_student_data()

    print('\nChoose an option:')
    print('1. Simple training and prediction')
    print('2. K-Fold cross validation')

    answer = input("\nEnter 1 or 2: ").strip()

    while answer not in ['1', '2']:
        print('\nInvalid selection. Please try again.')
        print('1. Simple training and prediction')
        print('2. K-Fold cross validation')
        answer = input("\nEnter 1 or 2: ").strip()

    if answer == '1':
        train_and_predict(student_dataset)
    elif answer == '2':
        k_fold_cross_validation(student_dataset)