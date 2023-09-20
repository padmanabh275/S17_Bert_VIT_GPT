from custom_models.transformers.gpt.utils import (
    BATCH_SIZE,
    BLOCK_SIZE,
    MAX_ITER,
    EVAL_INTER,
    get_batch,
    estimate_loss,
)
def train_gpt(MAX_ITER,train_data,val_data,optimizer,model):
    for step in range(MAX_ITER):
        
        # every EVAL_INTER evaluate the loss on train and val sets
        if step % EVAL_INTER == 0 or step == MAX_ITER - 1:
            loss_train = estimate_loss(
                data=train_data, model=model, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE
            )
            loss_val = estimate_loss(
                data=val_data, model=model, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE
            )
            print("step {:10} | train loss {:6.4f} | val loss {:6.4f}".format(step, loss_train, loss_val))

        # sample a batch of data
        xb, yb = get_batch(data=train_data, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE)
        logits, loss = model.forward(xb, yb)
        # zero_grad() method sets the gradients of all parameters in the optimizer to zero
        optimizer.zero_grad(set_to_none=True)
        # backward() method on the loss variable calculates the gradients 
        # of the loss with respect to the model's parameters.
        loss.backward()
        # step() method on the optimizer updates the model's parameters 
        # using the calculated gradients, in order to minimize the loss.
        optimizer.step()
    return model