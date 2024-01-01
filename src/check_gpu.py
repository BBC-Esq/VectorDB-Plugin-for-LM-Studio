import sys
from PySide6.QtWidgets import QApplication, QMessageBox

try:
    import torch
except ImportError:
    def display_info():
        app = QApplication(sys.argv)
        msg_box = QMessageBox(QMessageBox.Information, "PyTorch Not Installed", "PyTorch is not installed on this system.")
        msg_box.exec()

else:
    def check_bitsandbytes():
        try:
            import bitsandbytes as bnb
            p = torch.nn.Parameter(torch.rand(10, 10).cuda())
            a = torch.rand(10, 10).cuda()

            p1 = p.data.sum().item()

            adam = bnb.optim.Adam([p])

            out = a * p
            loss = out.sum()
            loss.backward()
            adam.step()

            p2 = p.data.sum().item()

            assert p1 != p2
            return "SUCCESS!\nInstallation of bitsandbytes was successful!"
        except ImportError:
            return "bitsandbytes is not installed."
        except AssertionError:
            return "bitsandbytes is installed, but the installation seems incorrect."
        except Exception as e:
            return f"An error occurred while checking bitsandbytes: {e}"

    def display_info():
        app = QApplication(sys.argv)
        info_message = ""

        if torch.cuda.is_available():
            info_message += "CUDA is available!\n"
            info_message += "CUDA version: {}\n\n".format(torch.version.cuda)
        else:
            info_message += "CUDA is not available.\n\n"

        if torch.backends.mps.is_available():
            info_message += "Metal/MPS is available!\n\n"
        else:
            info_message += "Metal/MPS is not available.\n\n"
            if not torch.backends.mps.is_built():
                info_message += "MPS not available because the current PyTorch install was not built with MPS enabled.\n\n"
            else:
                info_message += "MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.\n\n"

        info_message += "If you want to check the version of Metal and MPS on your macOS device, you can go to \"About This Mac\" -> \"System Report\" -> \"Graphics/Displays\" and look for information related to Metal and MPS.\n\n"

        if torch.version.hip is not None:
            info_message += "ROCm is available!\n"
            info_message += "ROCm version: {}\n".format(torch.version.hip)
        else:
            info_message += "ROCm is not available.\n"

        # Check for bitsandbytes
        bitsandbytes_message = check_bitsandbytes()
        info_message += "\n" + bitsandbytes_message

        msg_box = QMessageBox(QMessageBox.Information, "GPU Acceleration and Library Check", info_message)
        msg_box.exec()

if __name__ == "__main__":
    display_info()
