import numpy as np


class DataCleaner(object):
    def clean_data(self, df):
        # Replacing values to fit into Adaline classifier
        df.loc[df['Survived'] == 0, 'Survived'] = -1

        # Filling missing age data with average age
        df['Age'] = df['Age'].fillna(df['Age'].mean())

        # Array of columns that are needed to replace columns with categorical data
        columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Male', 'Female',
                   'Embarked_C', 'Embarked_S', 'Embarked_Q']

        # Filling new columns with zeros
        df = self.fill_with_zeros(df, columns)

        # Inserting actual values in corresponding columns
        df.loc[df['Sex'] == 'male', 'Male'] = 1
        df.loc[df['Sex'] == 'female', 'Female'] = 1
        df.loc[df['Pclass'] == 1, 'Pclass_1'] = 1
        df.loc[df['Pclass'] == 2, 'Pclass_2'] = 1
        df.loc[df['Pclass'] == 3, 'Pclass_3'] = 1
        df.loc[df['Embarked'] == 'C', 'Embarked_C'] = 1
        df.loc[df['Embarked'] == 'S', 'Embarked_S'] = 1
        df.loc[df['Embarked'] == 'Q', 'Embarked_Q'] = 1

        # Dropping "unimportant" data from dataset
        df = df.drop(['PassengerId', 'Pclass', 'Name', 'Sex',
                      'Ticket', 'Cabin', 'Embarked'], axis=1)

        return df

    def fill_with_zeros(self, df, columns):
        for column in columns:
            df[column] = np.zeros(df.shape[0])
        return df
