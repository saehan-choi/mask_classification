import matplotlib.pyplot as plt




train_loss = [1.1493326421634693, 0.1334668236066105, 0.053441807401478754, 0.025889241975238064, 0.01839026292328357, 0.014633507617634134, 0.011927883285032193, 0.00968001738047819, 0.007828532748145864, 0.008701597157638397, 0.006676730565968697, 0.006143428674856775, 0.005525282156072172, 0.008570094662572728, 0.015181359955793209, 0.028092954839479194, 0.01935118782346418, 0.019905316299745734, 0.014003630984044039, 0.0091963184781734]
val_loss = [0.7102333868582418, 0.5494812190063356, 0.5329090251228003, 0.5945234294951431, 0.6337437426950049, 0.6804414653402614, 0.6812154914919786, 0.7160042349747785, 0.6815915855073084, 0.7411141968618228, 0.8183684361731912, 0.8411362045349215, 0.9003951716024106, 0.8261224544306439, 0.9376741121193086, 0.7382441953176588, 0.6574002014370415, 0.6328834757091493, 0.7163379554673442, 0.6859114220761877]

# x = list(range(1,21))



plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('efficientnetb0 Loss test')

plt.xticks(range(0,22,2))

plt.plot(train_loss, label='train Loss')
plt.plot(val_loss,   label='val Loss')
plt.legend()
plt.show()

