# LoRA Low-Rank Adaptaion of Large Language Models

[LoRA](https://arxiv.org/abs/2106.09685) is a training method that accelerates the training of large language models while consuming less memory.
It adds pairs of rank-decomposition weight matrices (Called **Update matrices**) to existing weights, and only trains those newly added added weights.

It has some addvantages : 

* Previos pretrained weights are kept frozen so the model is not as prone to [catastrophic forgetting](https://www.pnas.org/doi/10.1073/pnas.1611835114).
* Rank-decomposition matrices have significantly fewer parameters than the original model, which means that trained LoRA weights are easily portable.

notice the following image : 

<img src="https://siddharthsharma1.files.wordpress.com/2023/04/image-31.png" width=50% height=50%>

let's say that m=100 , n=100 and k=5 (k refers to the maximum rank of matrix) then the original matrix size is 100 * 100 = 10000 which means 10000 parameters in that one.

But after using Rank-decomposition matrices you now have 100 * 5 = 500 and 5 * 100 = 500 which means 500 + 500 = 1000 parameters and that is a huge improvement.
* LoRA matrices are generally added to the attention layers of the original model. 
* The greater memory-efficiency allows you to run fine-tuning on consumer GPUs like the Tesla T4, RTX 3080 or even the RTX 2080 Ti! GPUs like the T4 are free and readily accessible in Kaggle or Google Colab notebooks.

