"""Training routine for TF."""
import os
import time
import numpy as np
from datetime import datetime


def crop_center(img, crop_size):
    """Center crop images."""
    x, y = img.shape[:2]
    cx, cy = crop_size
    startx = x // 2 - (cx // 2)
    starty = y // 2 - (cy // 2)
    return img[starty:starty + cy, startx:startx + cx]


def renormalize(img, max_value, min_value):
    """Normalize images to [0, 1]."""
    return (img - min_value) / (max_value - min_value)


def image_batcher(
        start,
        num_batches,
        images,
        labels,
        image_size,
        training_max,
        training_min):
    """Placeholder image/label batch loader."""
    for b in range(num_batches):
        next_image_batch = images[start:start + image_size[0]]
        image_stack = []
        label_stack = labels[start:start + image_size[0]]
        for patch in next_image_batch:
            # 4. Crop the center
            patch = crop_center(patch, image_size[1:3])
            # 5. Normalize to [0, 1]
            patch[patch > 1.] = 1.
            patch[patch < 0.] = 0.
            # 6. Add to list
            image_stack += [patch[None, :, :, :]]
        # Add dimensions and concatenate
        start += image_size[0]
        yield np.concatenate(image_stack, axis=0), label_stack, next_image_batch


def check_early_stop(
        perf_history,
        minimum_length=20,
        short_history=3,
        long_history=5,
        fail_function=np.less_equal):
    """
    Determine whether to stop early.

    We determine this point by comparing the performance from
    the last three epochs to the performance five epochs before those.
    Training runs for at least 20 epochs, and is terminated if all three
    of the last epochs show decreased performance or if
    800 epochs are reached.

    """
    if len(perf_history) < minimum_length:
        early_stop = False
    else:
        short_perf = perf_history[-short_history:]
        long_perf = perf_history[-long_history + short_history:short_history]
        short_check = fail_function(np.mean(long_perf), short_perf)
        if all(short_check):  # If we should stop
            early_stop = True
        else:
            early_stop = False

    return early_stop


def training_loop(
        config,
        neural_data,
        images,
        sess,
        train_op,
        summary_op,
        summary_writer,
        loss,
        checkpoint_dir,
        summary_dir,
        val_accuracy,
        train_accuracy,
        saver):
    """Run a training loop."""
    step, time_elapsed = 0, 0
    train_losses, train_accs, timesteps = {}, {}, {}
    val_losses, val_accs = {}, {}
    start_time = time.time()
    num_batches = np.floor(
        len(combined_files) / float(
            config.validation_batch)).astype(int)
    for image_batch, label_batch, file_batch in tqdm(
            image_batcher(
                start=0,
                num_batches=num_batches,
                images=combined_files,
                labels=combined_labels,
                config=config,
                training_max=training_max,
                training_min=training_min),
            total=num_batches):
        feed_dict = {
            images: image_batch
        }
        sc, tyh = sess.run(
            [scores, preds],
            feed_dict=feed_dict)
        dec_scores += [sc]
        yhat = np.append(yhat, tyh)
        y = np.append(y, label_batch)
        file_array = np.append(file_array, file_batch)








        train_vars = sess.run(train_dict.values())
        it_train_dict = {k: v for k, v in zip(
            train_dict.keys(), train_vars)}
        duration = time.time() - start_time
        train_losses[step] = it_train_dict['train_loss']
        train_accs[step] = it_train_dict['train_accuracy']
        timesteps[step] = duration
        assert not np.isnan(
            it_train_dict['train_loss']).any(),\
            'Model diverged with loss = NaN'
        if step % config.validation_iters == 0:
            it_val_acc = np.asarray([])
            it_val_loss = np.asarray([])
            for num_vals in range(config.num_validation_evals):
                # Validation accuracy as the average of n batches
                val_vars = sess.run(val_dict.values())
                it_val_dict = {k: v for k, v in zip(
                    val_dict.keys(), val_vars)}
                it_val_acc = np.append(
                    it_val_acc,
                    it_val_dict['val_accuracy'])
                it_val_loss = np.append(
                    it_val_loss,
                    it_val_dict['val_loss'])
            val_acc = it_val_acc.mean()
            val_lo = it_val_loss.mean()
            val_accs[step] = val_acc
            val_losses[step] = val_lo

            # Summaries
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)

            # Training status and validation accuracy
            format_str = (
                '%s: step %d, loss = %.2f (%.1f examples/sec; '
                '%.3f sec/batch) | Training accuracy = %s | '
                'Validation accuracy = %s | logdir = %s')
            print format_str % (
                datetime.now(),
                step,
                it_train_dict['train_loss'],
                config.batch_size / duration,
                float(duration),
                it_train_dict['train_accuracy'],
                val_acc,
                summary_dir)

            # Save the model checkpoint if it's the best yet
            if config.top_n_validation > 0:
                rep_idx = val_acc > val_accs
                if sum(rep_idx) > 0:
                    force_save = True
                    val_accs[np.argmax(rep_idx)] = val_acc
            else:
                force_save = True

            if force_save:
                ckpt_path = os.path.join(
                    checkpoint_dir,
                    'model_' + str(step) + '.ckpt')
                saver.save(
                    sess, ckpt_path, global_step=step)
                print 'Saved checkpoint to: %s' % ckpt_path
                force_save = False

                time_elapsed += float(duration)
            if config.early_stop:
                keys = np.sort([int(k) for k in val_accs.keys()])
                sorted_vals = np.asarray([val_accs[k] for k in keys])
                if check_early_stop(sorted_vals):
                    print 'Triggered an early stop.'
                    break
        else:
            # Training status
            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; '
                          '%.3f sec/batch) | Training accuracy = %s')
            print format_str % (
                datetime.now(),
                step,
                it_train_dict['train_loss'],
                config.batch_size / duration,
                float(duration),
                it_train_dict['train_accuracy'])

        # End iteration
        step += 1

    sess.close()
    return train_losses, val_losses, train_accs, val_accs, timesteps
