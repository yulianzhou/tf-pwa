import os.path
import sys

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_dir + "/..")


def main():
    """Calculate errors of a given set of parameters and their correlation coefficients."""
    import argparse

    parser = argparse.ArgumentParser(
        description="calculate errors of a given set of parameters and their correlation coefficients"
    )
    parser.add_argument(
        "--params", default="final_params.json", dest="params_file"
    )
    parser.add_argument("--input_file", default="data,phsp", dest="input_file")
    parser.add_argument("--save_file", default="amp.dat", dest="save_file")
    parser.add_argument(
        "--save_complex",
        action="store_true",
        default=False,
        dest="save_complex",
    )

    results = parser.parse_args()

    save_amp(
        params_file=results.params_file,
        input_file=results.input_file,
        save_file=results.save_file,
        save_complex=results.save_complex,
    )


def save_amp(params_file, input_file, save_file, save_complex):
    # import tf_pwa
    import numpy as np

    from tf_pwa.config_loader import ConfigLoader

    config = ConfigLoader("config.yml")
    config.set_params(params_file)
    if save_complex:
        amp = config.get_amplitude().decay_group.get_amp3
    else:
        amp = config.get_amplitude()

    for file_i in input_file.strip().split(","):
        data = config.get_data(file_i)
        for idx, data_i in enumerate(data):
            w = amp(data_i).numpy()
            if save_file.endswith(".npy"):
                np.save(f"{file_i}{idx}_" + save_file, w)
            else:
                w = np.reshape(w, (w.shape[0], -1))
                np.savetxt(f"{file_i}{idx}_" + save_file, w)


if __name__ == "__main__":
    main()
