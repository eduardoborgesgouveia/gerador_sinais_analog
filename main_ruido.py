import os
import numpy as np
from scipy.io.wavfile import write
import scipy as scipy
from plotly.subplots import make_subplots


class utils:

    def reconstruir_sinal(fft_real):
        """
        Reconstrói um sinal no tempo a partir da parte real da FFT.

        Argumentos:
        fft_real: A parte real da FFT do sinal.

        Retorna:
        sinal_reconstruido: O sinal reconstruído no tempo.
        """

        # Cria a parte imaginária da FFT a partir da parte real.
        fft_imag = np.zeros_like(fft_real)

        # A primeira e última frequência da parte imaginária são zero.
        fft_imag[0] = 0
        fft_imag[-1] = 0

        # As frequências negativas são o conjugado complexo das frequências positivas.
        for i in range(1, len(fft_real) // 2):
            fft_imag[i] = -fft_imag[-i]

        # Combina as partes real e imaginária para formar a FFT completa.
        fft_completa = fft_real + 1j * fft_imag

        # Realiza a inversa da FFT para obter o sinal no tempo.
        sinal_reconstruido = np.fft.ifft(fft_completa)

        return sinal_reconstruido

    def get_fft_visual_data_augumentation(x, range_freq=[0, 100], dt=2e-5):

        b = np.floor(len(x) / 2)
        b_completo = np.floor(len(x))
        c = len(x)
        df = 1 / (c * dt)
        try:
            x_amp = scipy.fft.fft(x.values)[: int(b)]
            x_amp_completo = scipy.fft.fft(x.values)[: int(b_completo)]
        except:
            x_amp = scipy.fft.fft(x)[: int(b)]
            x_amp_completo = scipy.fft.fft(x)[: int(b_completo)]

        x_amp = x_amp * 2 / c
        x_phase = np.angle(x_amp)
        x_amp = np.abs(x_amp)

        freq = np.arange(0, df * b, df)
        freq = freq[: int(b)]  # Frequency vector
        index_freq_completo = np.arange(0, df * b_completo, df)

        if range_freq is not None:
            x_amp = x_amp[(freq >= range_freq[0]) & (freq <= range_freq[1])]
            freq = freq[(freq >= range_freq[0]) & (freq <= range_freq[1])]

        return x_amp, freq, x_amp_completo, index_freq_completo


class gerador_sinais:

    def __init__(
        self,
        sample_rate,
        duration,
        lista_frequencias,
        lista_amplitudes,
        flag_pĺot_fft_final,
        debug=False,
    ):

        self.fs = sample_rate
        self.duration = duration
        self.frequencias = lista_frequencias
        self.amplitudes = lista_amplitudes
        self.flag_pĺot_fft_final = flag_pĺot_fft_final
        self.DEBUG = debug

        self.t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

        print("\n\n")
        print(" ------------- GERADOR ------------- ")
        print("")
        print(f"Frequencia de amostragem: {self.fs} Hz")
        print(f"Duração do sinal: {self.duration} segundos")
        print("Frequencias [Hz]: ", self.frequencias)
        print("Amplitudes: ", self.amplitudes)
        if self.DEBUG:
            print(
                "WARNING: A flag de debug está ativa. A execução do script gerará vários gráficos."
            )
            print("")
            print(
                " - Caso não queira gerar os gráficos de debug, altere o parametro 'DEBUG' para False. \n\n"
            )
        print("")
        if self.flag_pĺot_fft_final:
            print(
                "WARNING: A flag de plot da FFT está ativa. Será gerado um gráfico da DFT do sinal final."
            )
            print("")
            print(
                " - Caso não queira gerar o gráfico da DFT do sinal final, altere o parametro 'flag_pĺot_fft_final' para False. \n\n"
            )

        nyquist = self.fs / max(lista_frequencias)
        if nyquist < 2.56:
            print(
                f"WARNING: A frequencia de amostragem é de {self.fs} Hz enquanto você está gerando um sinal com frequencia de {max(lista_frequencias)} Hz."
            )
            print(
                f" - você deve aumentar a frequencia de amostragem para no mínimo {2.56*max(lista_frequencias)} Hz \n\n"
            )

    def gerar_sinal(self):
        print(" ------------- INICIO PROCESSO GERAR SINAL  ------------- \n\n")
        # Gerando a senoide
        fft_total = []
        for i in range(len(self.frequencias)):
            frequency = self.frequencias[i]
            amplitude = self.amplitudes[i]
            sine_wave = amplitude * np.sin(2 * np.pi * frequency * self.t)

            # Calculando a FFT da senoide
            yf, xf, yf_completo, xf_completo = utils.get_fft_visual_data_augumentation(
                sine_wave, range_freq=None, dt=1 / self.fs
            )

            if i == 0:
                fft_total = yf_completo
            else:
                fft_total += yf_completo

            if self.DEBUG:
                # # Plotando a fft
                fig = make_subplots(rows=1, cols=1)
                fig.add_scatter(
                    x=xf, y=np.abs(fft_total), mode="lines", name="FFT", row=1, col=1
                )
                fig.update_layout(title_text=f"FFT da Senoide {frequency} Hz")
                fig.show()

        if self.DEBUG:
            # Plotando a FFT
            fig = make_subplots(rows=1, cols=1)
            fig.add_scatter(
                x=xf_completo[: int(len(fft_total) / 2)],
                y=np.abs(fft_total)[: int(len(fft_total) / 2)],
                mode="lines",
                name="FFT",
                row=1,
                col=1,
            )
            fig.update_layout(title_text="FFT da Senoide")
            fig.show()

        # reconstruindo o sinal no tempo
        sine_wave = utils.reconstruir_sinal(fft_total)

        self.sine_wave = sine_wave

        # fazendo a fft do sinal reconstruido
        yf, xf, yf_completo, xf_completo = utils.get_fft_visual_data_augumentation(
            sine_wave, range_freq=None, dt=1 / self.fs
        )

        self.fft_sine_wave = (xf, yf)

        if self.flag_pĺot_fft_final:
            # Plotando a fft
            fig = make_subplots(rows=1, cols=1)
            fig.add_scatter(x=xf, y=np.abs(yf), mode="lines", name="FFT", row=1, col=1)
            fig.update_layout(title_text=f"FFT do Sinal Reconstruido")
            fig.show()

        print(" ------------- SINAL GERADO ------------- \n\n")

    def salvar_sinal(self, path):

        # Normalizando a senoide para valores entre -32767 e 32767 (16-bit PCM)
        audio_data = np.int16(self.sine_wave * 32767)

        # Salvando o arquivo de áudio (formato WAV)
        output_file = path
        write(output_file, self.fs, audio_data)

        print(
            f" ------------- Arquivo de áudio '{output_file}' criado com sucesso. -------------  \n\n"
        )


class gerador_ruido_branco:
    def __init__(
        self,
        sample_rate,
        duration,
        mean=0,
        std=1,
        type="white",
        flag_pĺot_fft_final=False,
    ):

        self.fs = sample_rate
        self.duration = duration

        self.flag_pĺot_fft_final = flag_pĺot_fft_final
        self.mean = mean
        self.std = std
        self.type = type
        self.t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

        print("\n\n")
        print(" ------------- GERADOR RUIDO BRANCO ------------- ")
        print("")
        print(f"Frequencia de amostragem: {self.fs} Hz")
        print(f"Duração do sinal: {self.duration} segundos")

    def gerar_ruido(self):
        print(" ------------- INICIO PROCESSO GERAR RUIDO  ------------- \n\n")
        # Gerando o ruido
        if self.type == "white":
            noise = np.random.normal(self.mean, self.std, len(self.t))
        else:
            print("Tipo de ruido não reconhecido. Usando ruido branco.")
            noise = np.random.normal(self.mean, self.std, len(self.t))

        # Calculando a FFT do ruido
        yf, xf, yf_completo, xf_completo = utils.get_fft_visual_data_augumentation(
            noise, range_freq=None, dt=1 / self.fs
        )

        # Plotando a fft
        fig = make_subplots(rows=1, cols=1)
        fig.add_scatter(x=xf, y=np.abs(yf), mode="lines", name="FFT", row=1, col=1)
        fig.update_layout(title_text=f"FFT do Ruido")
        fig.show()

        self.noise = noise

        print(" ------------- RUIDO GERADO ------------- \n\n")

    def salvar_ruido(self, path):

        # Normalizando o ruido para valores entre -32767 e 32767 (16-bit PCM)
        audio_data = np.int16(self.noise * 32767)

        # Salvando o arquivo de áudio (formato WAV)
        output_file = path
        write(output_file, self.fs, audio_data)

        print(
            f" ------------- Arquivo de áudio '{output_file}' criado com sucesso. -------------  \n\n"
        )


if __name__ == "__main__":
    # pegar o caminho do diretório atual
    path = os.getcwd()
    MAIN_PATH = path + "/senoide.wav"

    frequencia_amostragem = 100000  # Taxa de amostragem (samples por segundo)
    duracao = 10  # Duração do áudio em segundos

    frequencias = [1000, 3000, 10000]
    amplitudes = [1, 0.2, 0.01]

    gs = gerador_sinais(
        sample_rate=frequencia_amostragem,
        duration=duracao,
        lista_frequencias=frequencias,
        lista_amplitudes=amplitudes,
        flag_pĺot_fft_final=True,
        debug=False,
    )

    gs.gerar_sinal()

    gs.salvar_sinal(MAIN_PATH)

    MAIN_PATH = path + "/ruido.wav"

    mean = 0
    std = 1
    type = "white"

    gr = gerador_ruido_branco(
        sample_rate=frequencia_amostragem,
        duration=duracao,
        mean=mean,
        std=std,
        type=type,
        flag_pĺot_fft_final=True,
    )

    gr.gerar_ruido()

    gr.salvar_ruido(MAIN_PATH)
