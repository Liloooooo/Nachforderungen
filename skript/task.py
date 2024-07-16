from trainer import Trainer, data_reader
import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-dir', type=str, help='directory that contains training files named "Daten_Arbeitsprobe_DS.csv" ', required=True)
args = parser.parse_args()

#Training
if __name__=='__main__':
    input_path=args.dir
    filename="Daten_Arbeitsprobe_DS.csv"
    features_train, target_train=data_reader(path=input_path, filename=filename)
    rf_trainer=Trainer(features_train, target_train)
    trained_model=rf_trainer.fit()

    model_save="rf.pkl"
    with open(model_save, 'wb') as file:  
        pickle.dump(model_save, file)
    print(f'model saved as {model_save}')

