import pandas as pd
import subprocess
from io import StringIO
from time import sleep


df = None

while True:
    result = subprocess.check_output(
        ["kaggle", "competitions", "submissions", "-c", "espejito-espejito-inf577", "--csv"],
        text=True
    )

    df = pd.read_csv(StringIO(result))

    submission_status = df.iloc[0]["status"]

    if submission_status == "SubmissionStatus.COMPLETE":
        break
    else:
        sleep(0.5)

last_score = df.iloc[0]["publicScore"]
best_score = df["publicScore"].max()

if last_score == best_score:
    print("¡Último envío es el mejor hasta ahora!")
    second_best_score = df[df["publicScore"] < best_score]["publicScore"].max()

    print(f"Nuevo mejor score:   {best_score:.4f}")
    if second_best_score is not None:
        print(f"Segundo mejor score: {second_best_score:.4f}")
        print(f"Mejora de:           {best_score - second_best_score:.4f}")
else:
    print(f"Último score: {last_score:.4f}")
    print(f"Mejor score:  {best_score:.4f}")

    # Print the difference Green 
    print(f"Diferencia:   {best_score - last_score:.4f}")