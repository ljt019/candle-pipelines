# Running Tests

Because the integration tests require actually loading the model into memory and running inference you can't run them. You can't run anything that requires actual inference because the machine you run on doesn't have access to a powerful enough cpu or a gpu at all. If you want to test any code that requires inference end your session and return a message to the user with what need to be ran/tested. The user will then run it on their machine and return the results to you so you can continue working.

# Candle Reference 

We don't depend on the local version but the entire candle repo is clone into './candle' so you can reference it's source code if you are ever confused whiel working with it or it's apis.