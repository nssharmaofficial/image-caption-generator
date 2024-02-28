import torch
import torch.utils.data
from config import Config
from dataset import (ImageCaptionDataset, get_data_loader, preprocessing_transforms)
from model import Decoder, Encoder, get_acc
from vocab import Vocab
import gc


if __name__ == '__main__':

    config = Config()

    print("Loading vocabulary...")
    vocab = Vocab()
    vocab.load_vocab(config.VOCAB_FILE)

    print("Creating ImageCaptionDataset...")
    train_data = ImageCaptionDataset('train_list.txt', vocab, 'images', transform=preprocessing_transforms())
    val_data = ImageCaptionDataset('val_list.txt', vocab, 'images', transform=preprocessing_transforms())

    print("Setting up data loaders...")
    train_loader = get_data_loader(train_data, batch_size=config.BATCH, pad_index=vocab.PADDING_INDEX)
    val_loader = get_data_loader(val_data, batch_size=config.BATCH, pad_index=vocab.PADDING_INDEX)

    print("Creating model...")
    image_encoder = Encoder(image_emb_dim=config.IMAGE_EMB_DIM,
                            device=config.DEVICE)
    emb_layer = torch.nn.Embedding(num_embeddings=config.VOCAB_SIZE,
                                   embedding_dim=config.WORD_EMB_DIM,
                                   padding_idx=vocab.PADDING_INDEX)
    image_decoder = Decoder(image_emb_dim=config.IMAGE_EMB_DIM,
                            word_emb_dim=config.WORD_EMB_DIM,
                            hidden_dim=config.HIDDEN_DIM,
                            num_layers=config.NUM_LAYER,
                            vocab_size=config.VOCAB_SIZE,
                            device=config.DEVICE)

    criterion = torch.nn.CrossEntropyLoss().requires_grad_(True)
    parameters = list(image_decoder.parameters()) + list(emb_layer.parameters()) + list(image_encoder.parameters())
    for param in parameters:
        param.requires_grad_(True)
    optimizer = torch.optim.Adam(params=parameters, lr=config.LR)

    image_encoder = image_encoder.to(config.DEVICE)
    emb_layer = emb_layer.to(config.DEVICE)
    image_decoder = image_decoder.to(config.DEVICE)

    print("Beginning Training")

    training_batch_losses = []
    training_batch_accuracies = []
    validation_batch_losses = []
    validation_batch_accuracies = []

    training_losses = []
    training_acc = []
    validation_losses = []
    validation_acc = []

    for epoch in range(0, config.EPOCHS):

        for i, batch in enumerate(train_loader):

            image_encoder.train()
            emb_layer.train()
            image_decoder.train()

            images_batch, captions_batch = batch[0].to(config.DEVICE), batch[1].to(config.DEVICE)
            # image_batch : (BATCH, 3, 224, 224)
            # captions_batch : (BATCH, SEQ_LENGTH)

            decoder_targets = captions_batch[:, :]
            decoder_inputs = captions_batch[:, :]
            mask = (decoder_targets != vocab.PADDING_INDEX).float()
            # all : (BATCH, SEQ_LENGTH)

            features = image_encoder.forward(images_batch)
            # features : (BATCH, IMAGE_EMB_DIM)
            features = features.unsqueeze(0)
            # features : (1, BATCH, IMAGE_EMB_DIM)

            t_loss = 0
            t_accuracy = 0

            emb_captions_batch = emb_layer.forward(decoder_inputs)
            # captions_batch : (BATCH, SEQ_LENGTH, WORD_EMB_DIM)

            emb_captions_batch = emb_captions_batch.permute(1, 0, 2)
            # captions_batch : (SEQ_LENGTH, BATCH, WORD_EMB_DIM)

            BATCH_SIZE = emb_captions_batch.shape[1]
            SEQ_LENGTH = emb_captions_batch.shape[0]

            features = features.repeat(SEQ_LENGTH, 1, 1)
            # features:  (1, BATCH, IMAGE_EMB_DIM) --> (SEQ_LENGTH, BATCH, IMAGE_EMB_DIM)

            hidden = image_decoder.hidden_state_0.repeat(1, BATCH_SIZE, 1)
            cell = image_decoder.cell_state_0.repeat(1, BATCH_SIZE, 1)
            # hidden and cell : (NUM_LAYER, 1, HIDDEN_DIM) --> (NUM_LAYER, BATCH, HIDDEN_DIM)

            output, (hidden, cell) = image_decoder.forward(emb_captions_batch, features, hidden, cell)
            # output : (SEQ_LENGTH, BATCH, VOCAB_SIZE)
            # hidden and cell : (NUM_LAYER, BATCH, HIDDEN_DIM)

            decoder_targets = decoder_targets.permute(1, 0)
            mask = mask.permute(1, 0)
            # all : (SEQ_LENGTH, BATCH)

            # sum up loss and accuracy for each word (!=pad) in sentence for each training example in batch
            for j in range(SEQ_LENGTH-1):
                t_loss += criterion(output[j, :, :], decoder_targets[j+1, :])*mask[j+1, :]
                t_accuracy += get_acc(output[j, :, :], decoder_targets[j+1, :])*mask[j+1, :]

            # loss summed up through batch divided by the total number of words (!=pad) in batch
            t_loss = t_loss.sum() / mask.sum().item()
            t_accuracy = t_accuracy.sum() / mask.sum().item()

            optimizer.zero_grad()
            t_loss.backward()
            optimizer.step()

            # Print stats every 100 iterations
            if i % 100 == 0:
                print("Epoch: [%d/%d], Step: [%d/%d], Loss: %.3f, Accuracy: %.3f " % (epoch+1,
                                                                                      config.EPOCHS,
                                                                                      i,
                                                                                      len(train_loader),
                                                                                      t_loss.item(),
                                                                                      t_accuracy*100))

            # store results for each batch
            t_loss = t_loss.to(torch.device("cpu"))
            t_accuracy = t_accuracy.to(torch.device("cpu"))
            training_batch_losses.append(t_loss)
            training_batch_accuracies.append(t_accuracy)

        # get the average results for each epoch
        training_loss_avg = sum(training_batch_losses) / len(training_batch_losses)
        training_acc_avg = sum(training_batch_accuracies) / len(training_batch_accuracies)
        training_losses.append(float(training_loss_avg.item()))
        training_acc.append(float(training_acc_avg.item()))

        torch.cuda.empty_cache()
        gc.collect()

        # print(training_losses, training_acc)

        for k, batch in enumerate(val_loader):

            image_encoder.eval()
            emb_layer.eval()
            image_decoder.eval()

            with torch.no_grad():

                image_batch, captions_batch = batch[0].to(config.DEVICE), batch[1].to(config.DEVICE)
                # image_batch : (BATCH, 3, 224, 224)
                # captions_batch : (BATCH, SEQ_LENGTH)

                decoder_targets = captions_batch[:, :]
                decoder_inputs = captions_batch[:, :]
                mask = (decoder_targets != vocab.PADDING_INDEX).float()
                # all : (BATCH, SEQ_LENGTH)

                features = image_encoder(image_batch)
                # features : (32, IMAGE_EMB_DIM)
                features = features.unsqueeze(0)
                # features : (1, BATCH, IMAGE_EMB_DIM)

                v_loss = 0
                v_accuracy = 0

                emb_captions_batch = emb_layer.forward(decoder_inputs)
                # captions_batch : (BATCH, SEQ_LENGTH, WORD_EMB_DIM)

                emb_captions_batch = emb_captions_batch.permute(1, 0, 2)
                # captions_batch : (SEQ_LENGTH, BATCH, WORD_EMB_DIM)

                BATCH_SIZE = emb_captions_batch.shape[1]
                SEQ_LENGTH = emb_captions_batch.shape[0]

                features = features.repeat(SEQ_LENGTH, 1, 1)
                # features:  (SEQ_LENGTH, BATCH, IMAGE_EMB_DIM)

                hidden = image_decoder.hidden_state_0.repeat(1, BATCH_SIZE, 1)
                cell = image_decoder.cell_state_0.repeat(1, BATCH_SIZE, 1)
                # hidden and cell : (NUM_LAYER, BATCH, HIDDEN_DIM)

                output, (hidden, cell) = image_decoder.forward(emb_captions_batch, features, hidden, cell)
                # output : (SEQ_LENGTH, BATCH, VOCAB_SIZE)
                # hidden and cell : (NUM_LAYER, BATCH, HIDDEN_DIM)

                decoder_targets = decoder_targets.permute(1, 0)
                mask = mask.permute(1, 0)
                # all : (SEQ_LENGTH, BATCH)

                # sum up loss and accuracy for each word (!=pad) in sentence for each training example in batch
                for j in range(SEQ_LENGTH-1):
                    v_loss += criterion(output[j, :, :], decoder_targets[j+1, :])*mask[j+1, :]
                    v_accuracy += get_acc(output[j, :, :], decoder_targets[j+1, :])*mask[j+1, :]

                # loss summed up through batch divided by the total number of words (!=pad) in batch
                v_loss = v_loss.sum() / mask.sum().item()
                v_accuracy = v_accuracy.sum() / mask.sum().item()

                # Print stats every 100 iterations
                if k % 100 == 0:
                    print("Epoch: [%d/%d], Step: [%d/%d], Loss: %.3f,  Accuracy: %.3f " % (epoch+1,
                                                                                           config.EPOCHS,
                                                                                           k,
                                                                                           len(val_loader),
                                                                                           v_loss.item(),
                                                                                           v_accuracy*100))
                # store results for each batch
                v_loss = v_loss.to(torch.device("cpu"))
                v_accuracy = v_accuracy.to(torch.device("cpu"))
                validation_batch_losses.append(v_loss)
                validation_batch_accuracies.append(v_accuracy)

        # get the average results for each epoch
        validation_loss_avg = sum(validation_batch_losses) / len(validation_batch_losses)
        validation_acc_avg = sum(validation_batch_accuracies) / len(validation_batch_accuracies)
        validation_losses.append(float(validation_loss_avg.item()))
        validation_acc.append(float(validation_acc_avg.item()))

        torch.cuda.empty_cache()
        gc.collect()

        # print(validation_losses, validation_acc)

        # save model after every epoch
        torch.save(image_encoder.state_dict(),
                   f"checkpoints/NEW_encoder-{config.BATCH}B-{config.HIDDEN_DIM}H-{config.NUM_LAYER}L-e{epoch+1}.pt")
        torch.save(emb_layer.state_dict(),
                   f"checkpoints/NEW_embeddings-{config.BATCH}B-{config.HIDDEN_DIM}H-{config.NUM_LAYER}L-e{epoch+1}.pt")
        torch.save(image_decoder.state_dict(),
                   f"checkpoints/NEW_decoder-{config.BATCH}B-{config.HIDDEN_DIM}H-{config.NUM_LAYER}L-e{epoch+1}.pt")

    from matplotlib import pyplot as plt

    plt.subplot(1, 2, 1)
    plt.plot(training_acc)
    plt.plot(validation_acc)
    plt.title('Accuracies vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])

    plt.subplot(1, 2, 2)
    plt.plot(training_losses)
    plt.plot(validation_losses)
    plt.title('Losses vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])

    plt.savefig(f'saved/NEW_{config.BATCH}B-{config.HIDDEN_DIM}H-{config.NUM_LAYER}L.jpg')
