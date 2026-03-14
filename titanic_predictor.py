
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)


df = sns.load_dataset('titanic')

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())


fig, axes = plt.subplots(1, 3, figsize=(14, 4)
                         
sns.barplot(data=df, x='sex', y='survived', ax=axes[0], palette='Set2')
axes[0].set_title('Survival rate by gender')
axes[0].set_ylabel('Survival rate')

sns.barplot(data=df, x='pclass', y='survived', ax=axes[1], palette='Set2')
axes[1].set_title('Survival rate by class')
axes[1].set_xlabel('Passenger class (1=first)')

df[df['survived'] == 1]['age'].dropna().hist(ax=axes[2], alpha=0.6, label='Survived', bins=20, color='steelblue')
df[df['survived'] == 0]['age'].dropna().hist(ax=axes[2], alpha=0.6, label='Died', bins=20, color='salmon')
axes[2].set_title('Age distribution')
axes[2].legend()

plt.tight_layout()
plt.savefig('exploration.png', dpi=120)
plt.show()
print("Saved: exploration.png")


features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
target = 'survived'

data = df[features + [target]].copy()

data['age'].fillna(data['age'].median(), inplace=True)
data['fare'].fillna(data['fare'].median(), inplace=True)
data['embarked'].fillna(data['embarked'].mode()[0], inplace=True)

le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])        
data['embarked'] = le.fit_transform(data['embarked'])

data['family_size'] = data['sibsp'] + data['parch'] + 1

data['is_alone'] = (data['family_size'] == 1).astype(int)

print("\nFeatures after engineering:")
print(data.head())
print("\nNo missing values?", data.isnull().sum().sum() == 0)


X = data.drop(target, axis=1)
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")


models = {
    'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cv = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    results[name] = {'model': model, 'accuracy': acc, 'cv_accuracy': cv, 'preds': preds}
    print(f"\n{name}")
    print(f"  Test accuracy:  {acc:.3f}")
    print(f"  CV accuracy:    {cv:.3f}")
    print(classification_report(y_test, preds, target_names=['Died', 'Survived']))


best_name = max(results, key=lambda k: results[k]['accuracy'])
best = results[best_name]

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, best['preds'],
    display_labels=['Died', 'Survived'],
    cmap='Blues', ax=ax
)
ax.set_title(f'Confusion matrix — {best_name}')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=120)
plt.show()
print("Saved: confusion_matrix.png")


rf = results['Random Forest']['model']
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values()

plt.figure(figsize=(7, 4))
importances.plot(kind='barh', color='steelblue')
plt.title('Feature importance — Random Forest')
plt.xlabel('Importance score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=120)
plt.show()
print("Saved: feature_importance.png")


def predict_survival(pclass, sex, age, sibsp, parch, fare, embarked):
    """
    Predict survival for a custom passenger.
    sex: 'male' or 'female'
    embarked: 'S', 'C', or 'Q'
    """
    embarked_map = {'C': 0, 'Q': 1, 'S': 2}
    sex_val = 1 if sex == 'male' else 0
    emb_val = embarked_map.get(embarked, 2)
    family_size = sibsp + parch + 1
    is_alone = int(family_size == 1)

    row = pd.DataFrame([{
        'pclass': pclass, 'sex': sex_val, 'age': age,
        'sibsp': sibsp, 'parch': parch, 'fare': fare,
        'embarked': emb_val, 'family_size': family_size, 'is_alone': is_alone
    }])

    pred = rf.predict(row)[0]
    prob = rf.predict_proba(row)[0][1]
    outcome = "SURVIVED" if pred == 1 else "DIED"
    print(f"Prediction: {outcome}  (survival probability: {prob:.1%})")
    return pred, prob


print("\n--- Custom predictions ---")
predict_survival(pclass=1, sex='female', age=28, sibsp=0, parch=0, fare=100, embarked='S')
predict_survival(pclass=3, sex='male',   age=22, sibsp=1, parch=0, fare=7,   embarked='S')
predict_survival(pclass=2, sex='female', age=35, sibsp=0, parch=1, fare=30,  embarked='C')


print("\n=== Project complete! ===")
print("Next steps:")
print("  1. Try tuning n_estimators or max_depth on Random Forest")
print("  2. Add a 'title' feature extracted from passenger names")
print("  3. Try XGBoost for a likely accuracy boost")
print("  4. Push to GitHub with a clear README")
