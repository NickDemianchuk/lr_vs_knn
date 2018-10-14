class DataCleaner(object):
    def clean_data(self, df):
        # Replacing values to match class labels
        df.loc[df['Survived'] == 0, 'Survived'] = -1

        # Filling missing age data with average age
        df['Age'] = df['Age'].fillna(df['Age'].mean())

        # Replacing categorical values by integers
        df.loc[df['Sex'] == 'male', 'Sex'] = 0
        df.loc[df['Sex'] == 'female', 'Sex'] = 1
        df.loc[df['Embarked'] == 'C', 'Embarked'] = 1
        df.loc[df['Embarked'] == 'S', 'Embarked'] = 2
        df.loc[df['Embarked'] == 'Q', 'Embarked'] = 3

        # Replacing missing values with 0
        df['Embarked'] = df['Embarked'].fillna(0)

        # Dropping "unimportant" data from dataset
        df = df.drop(['PassengerId', 'Name',
                      'Ticket', 'Cabin'], axis=1)

        return df