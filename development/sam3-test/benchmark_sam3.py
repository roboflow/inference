from sam3 import build_sam3_image_model
import torch
import time


import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

torch.set_grad_enabled(False)

checkpoint_path = "/tmp/cache/sam3/default/sam3_prod_v12_interactive_5box_image_only.pt"
bpe_path = "/tmp/cache/sam3/default/bpe_simple_vocab_16e6.txt.gz"


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timeit(fn):
    _sync()
    start = time.perf_counter()
    result = fn()
    _sync()
    end = time.perf_counter()
    return result, (end - start)


def run_once():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def _build():
        return build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=checkpoint_path,
            device=device,
            eval_mode=True,
        )

    model, t_build = _timeit(_build)

    model.compile_model = True
    # _, compile_time = _timeit(lambda: model._compile_model())
    # _, compile_time = _timeit(lambda: model.warm_up_compilation())


    image_path = "/home/hansent/images/traffic.jpg"

    inference_state, t_init = _timeit(lambda: model.init_state(image_path))
    _, t_reset = _timeit(lambda: model.reset_state(inference_state))
    _, t_infer = _timeit(
        lambda: model.add_prompt(
            inference_state,
            text_str="floor",
            output_prob_thresh=0.5,
            instance_prompt=False,
        )
    )

    print("\nSAM3 Benchmark (default config)")
    print(f"- build_model: {t_build*1000:.2f} ms ({t_build:.4f} s)")
    #print(f"- warm_up_compilation: {compile_time*1000:.2f} ms ({compile_time:.4f} s)")
    print(f"- init_state:  {t_init*1000:.2f} ms ({t_init:.4f} s)")
    print(f"- reset_state: {t_reset*1000:.2f} ms ({t_reset:.4f} s)")
    print(f"- add_prompt:  {t_infer*1000:.2f} ms ({t_infer:.4f} s)")

    # Second image benchmark
    image_path2 = "/home/hansent/images/test.jpg"
    inference_state2, t_init2 = _timeit(lambda: model.init_state(image_path2))
    _, t_reset2 = _timeit(lambda: model.reset_state(inference_state2))
    _, t_infer2 = _timeit(
        lambda: model.add_prompt(
            inference_state2,
            text_str="floor",
            output_prob_thresh=0.5,
            instance_prompt=False,
        )
    )

    print("\nSAM3 Benchmark (second image: test.jpg)")
    print(f"- init_state:  {t_init2*1000:.2f} ms ({t_init2:.4f} s)")
    print(f"- reset_state: {t_reset2*1000:.2f} ms ({t_reset2:.4f} s)")
    print(f"- add_prompt:  {t_infer2*1000:.2f} ms ({t_infer2:.4f} s)")


def main():
    run_once()


if __name__ == "__main__":
    main()