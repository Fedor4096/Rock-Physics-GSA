import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import plotly.offline
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import seaborn as sns
from bisect import bisect_left
import numba
import random

pio.renderers.default = "browser"

s_t = time.time()

# Рассматривается случай расчёта по методу GSA для двухкомпонетной горной породы: минериальная матрица и флюид

# Задаются начальные параметры для матрицы: параметры Ламе, концетрация и аспектные отношения эллипсоида
k_matrix: float = 45.0
mu_matrix: float = 20.0
v_matrix: float =  0.996
a1_matrix: float = 1.0
a2_matrix: float = 1.0
a3_matrix: float = 1.0

# Задаются аналогичные начальные параметры для флюида
k_fluid: float = 2.25
mu_fluid: float = 0.0
v_fluid: float =  1 - v_matrix
a1_fluid: float = 1000.0
a2_fluid: float = 1000.0
a3_fluid: float = 1.0

# Определить параметр связанности пор
f: float = 0.8

# Задать параметры для расчёта сетки интегрирования для матрицы и для флюида отдельно
# Указывается начальное и конечное значения интервала для получения точек интегрирования,
# а также третим параметром - количество точек, равномерно распределённых на нём
range_tetha_matrix = [[0.0,np.pi,200]]
range_phi_matrix = [[0,2*np.pi,200]]

range_tetha_fluid = [[0.0,1.555,100],[1.555,1.59,100],[1.59,np.pi,100]]
range_phi_fluid = [[0,2*np.pi,60]]

# Переход от матричной записи в нотации Фойгта к тензорному виду
# Параметр "index" отвечает за деление необходимых элементов матрицы
# для работы с тензором податливости. Для случая тензора упругости
# никакие дополнительные коэффициенты не нужны
@numba.njit()
def convert_voigt_to_full_stiffness_matrix(voigt_matrix, index) -> np.ndarray:
    
    def full_to_voigt_index(i: int, j: int) -> int:
        if i == j:
            return i
        return (6-i-j)

    if index == True:
        for i in range(6):
            for j in range(6):
                if i > 2:
                    voigt_matrix[i,j] /= 2
                if j > 2:
                    voigt_matrix[i,j] /= 2
    full_matrix = np.zeros((3, 3, 3, 3), dtype=float)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    voigt_i = full_to_voigt_index(i, j)
                    voigt_j = full_to_voigt_index(k, l)
                    full_matrix[i, j, k, l] = voigt_matrix[voigt_i, voigt_j]

    return full_matrix

# Переход от тензорного вида к матричной записи в нотации Фойгта
# Параметр "index" отвечает за умножение необходимых элементов матрицы
# для работы с тензором податливости. Для случая тензора упругости
# никакие дополнительные коэффициенты не нужны
@numba.njit()
def convert_full_stiffness_matrix_to_voigt(C, index):

    voigt_indices = [(0, 0), (1, 1), (2, 2), (1, 2), (2, 0), (0, 1)]
    voigt_tensor = np.zeros((6, 6), dtype=float)

    for i in range(6):
        for j in range(6):
            k, l = voigt_indices[i]
            m, n = voigt_indices[j]
            voigt_tensor[i, j] = C[k, l, m, n]

    if index == True:
        for i in range(6):
            for j in range(6):
                if i > 2:
                    voigt_tensor[i,j] *= 2
                if j > 2:
                    voigt_tensor[i,j] *= 2
    
    return voigt_tensor

# Расчёт тензора упругости на основе параметров Ламе для изотропного случая
@numba.njit()
def calculate_Cklmn_from_k_mu(k_rock: float, mu_rock: float)-> np.ndarray:

    lambda_ = k_rock - (2 * mu_rock / 3)
    c11 = lambda_ + (2 * mu_rock)
    c12 = lambda_
    c44 = mu_rock

    C6x6 = np.zeros((6, 6), dtype=float)

    C6x6[0, 0] = c11
    C6x6[0, 1] = c12
    C6x6[0, 2] = c12
    C6x6[1, 0] = c12
    C6x6[1, 1] = c11
    C6x6[1, 2] = c12
    C6x6[2, 0] = c12
    C6x6[2, 1] = c12
    C6x6[2, 2] = c11
    C6x6[3, 3] = c44
    C6x6[4, 4] = c44
    C6x6[5, 5] = c44

    Cklmn = convert_voigt_to_full_stiffness_matrix(C6x6, False)

    return Cklmn

# Вывод в консоль всех компонент тензора 4 ранга
def print_full_tensor(a):
    for k in range(3):
        for m in range(3):
            for l in range(3):
                for n in range(3):
                    print(f"{k+1} {m+1} {l+1} {n+1} \t {a[k,m,l,n]:.8}")

# Вывод в консоль компонент тензора 4 ранга в матричном виде нотации Фойгта
def print_voigt(a):
    for i in range(6):
        for j in range(6):
            print(f"{a[i,j]:.6}\t", end='')
        print("\n")
    print("\n")

# Сохранить в отдельном PNG-файле все компоненты тензора 4 ранга
def draw_full_stiffness_matrix_representation(a, name):
    v_min=0
    v_max = np.max(a)
    if np.all(a==0) != True:
        v_min = np.min(a[np.nonzero(a)])
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))
    axes = axes.flatten()
    sns.set(style='white')
    counter = 0
    for i in range(3):
        for j in range(3):
            if np.all(a==0) != True:
                sns.heatmap(a[i,j], annot=True, linewidth=0.5, xticklabels=False, yticklabels=False,vmin=v_min,vmax=v_max,cbar=False,
                            linecolor='gray',square=True,mask=(a[i][j]==0),fmt='.6F',cmap='coolwarm',annot_kws={"size":8},ax=axes[counter]).set_facecolor('white')
            else:
                sns.heatmap(a[i,j], annot=True, linewidth=0.5, xticklabels=False, yticklabels=False,vmin=v_min,vmax=v_max,cbar=False,
                            linecolor='gray',square=True,fmt='.6F',cmap='coolwarm',annot_kws={"size":8},ax=axes[counter]).set_facecolor('white')
            ax_s = axes[counter].axis()
            rec = axes[counter].add_patch(Rectangle((ax_s[0],ax_s[2]),(ax_s[1]-ax_s[0]),(ax_s[3]-ax_s[2]),fill=False,lw=3, edgecolor='black'))
            rec.set_clip_on(False)
            counter += 1
    for ax in axes:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    counter = 0
    for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        if a[i,j,k,l] == 0.0:
                            axes[counter].text(l+0.5, k+0.5, f'0.0', fontdict=dict(ha='center',  va='center',color='black', fontsize=8))
                        axes[counter].text(l+0.5, k+0.2, f'{i+1}|{j+1}|{k+1}|{l+1}', fontdict=dict(ha='center',  va='center',color='grey', fontsize=6))
                counter += 1
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f"{name}.png", dpi=300)

# Сохранить в отдельном PNG-файле компоненты тензора 4 ранга в матричном виде нотации Фойгта
def draw_voigt_representation(a, name):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    sns.set(style='white')
    voigt_indices = [(0, 0), (1, 1), (2, 2), (1, 2), (2, 0), (0, 1)]
    if np.all(a==0) == True:
        sns.heatmap(a, annot=False, linewidth=0.5, xticklabels=False, yticklabels=False,vmin=0,vmax=0,cbar=False,
                    linecolor='gray',square=True,mask=(a==0),fmt='.6F',cmap='coolwarm',ax=ax1).set_facecolor('white')
        for i in range(6):
            for j in range(6):
                k, l = voigt_indices[i]
                m, n = voigt_indices[j]
                ax1.text(j+0.5, i+0.5, '0.0', fontdict=dict(ha='center',  va='center',color='black', fontsize=8))
                ax1.text(j+0.5, i+0.2, f'{k+1}|{l+1}|{m+1}|{n+1}', fontdict=dict(ha='center',  va='center',color='grey', fontsize=6))

    else:
        sns.heatmap(a, annot=True, linewidth=0.5, xticklabels=False, yticklabels=False,vmin=np.min(a[np.nonzero(a)]),vmax=np.max(a),cbar=False,
                    linecolor='gray',square=True,mask=(a==0),fmt='.6F',cmap='coolwarm',annot_kws={"size":8},ax=ax1).set_facecolor('white')
        for i in range(6):
            for j in range(6):
                k, l = voigt_indices[i]
                m, n = voigt_indices[j]
                if a[i,j] == 0.0:
                    ax1.text(j+0.5, i+0.5, f'{a[i,j]}', fontdict=dict(ha='center',  va='center',color='black', fontsize=8))
                ax1.text(j+0.5, i+0.2, f'{k+1}|{l+1}|{m+1}|{n+1}', fontdict=dict(ha='center',  va='center',color='grey', fontsize=6))
    ax_s = ax1.axis()
    rec = ax1.add_patch(Rectangle((ax_s[0],ax_s[2]),(ax_s[1]-ax_s[0]),(ax_s[3]-ax_s[2]),fill=False,lw=3, edgecolor='black'))
    rec.set_clip_on(False)
    plt.savefig(f"{name}.png", dpi=300)

# Расчёт тензоров упругости для матрицы и флюида по заданным параметрам Ламе
C_matrix = calculate_Cklmn_from_k_mu(k_matrix, mu_matrix)
C_fluid = calculate_Cklmn_from_k_mu(k_fluid, mu_fluid)

# Расчёт тензора упругости для тела сравнения
Cklmn = (1-f)*C_matrix+f*C_fluid

# Расчёт узлов сетки интегрирования отдельно для матрицы и флюида
def get_axes_for_all_areas(range_tetha_matrix, range_phi_matrix, range_tetha_fluid, range_phi_fluid):
    
    tetha_count = 0
    phi_count = 0

    for i in range_tetha_matrix:
        tetha_count += i[2]
    for i in range_phi_matrix:
        phi_count += i[2]

    tetha_matrix = np.zeros((tetha_count), dtype=float)
    phi_matrix = np.zeros((phi_count), dtype=float)

    counter = 0
    for i in range_tetha_matrix:
        tetha_matrix[counter:counter+i[2]] = np.linspace(i[0],i[1], i[2])
        counter += i[2]
    counter = 0
    for i in range_phi_matrix:
        phi_matrix[counter:counter+i[2]] = np.linspace(i[0],i[1], i[2])
        counter += i[2]

    #////////////////////////////////////////////////////////////////////
    
    tetha_count = 0
    phi_count = 0

    for i in range_tetha_fluid:
        tetha_count += i[2]
    for i in range_phi_fluid:
        phi_count += i[2]
    
    tetha_fluid = np.zeros((tetha_count), dtype=float)
    phi_fluid = np.zeros((phi_count), dtype=float)

    counter = 0
    for i in range_tetha_fluid:
        tetha_fluid[counter:counter+i[2]] = np.linspace(i[0],i[1], i[2])
        counter += i[2]
    counter = 0
    for i in range_phi_fluid:
        phi_fluid[counter:counter+i[2]] = np.linspace(i[0],i[1], i[2])
        counter += i[2]

    return tetha_matrix, phi_matrix, tetha_fluid, phi_fluid

tetha_matrix, phi_matrix, tetha_fluid, phi_fluid = get_axes_for_all_areas(range_tetha_matrix, range_phi_matrix, range_tetha_fluid, range_phi_fluid)

# Расчёт вспомогательных значений n, входящих в форулы для матрицы Лямбда
# и подынтегральной функции одной комопненты несимметризованного тензора a_not_sym
@numba.njit()
def calculate_Nmn(tetta, phi, a1, a2 ,a3, n, m):

    n_all = np.zeros(3)
    n_all[0] = np.sin(tetta) * np.cos(phi) / a1
    n_all[1] = np.sin(tetta) * np.sin(phi) / a2
    n_all[2] = np.cos(tetta) / a3
    
    return n_all[n] * n_all[m]

# Расчёт одной инвертированной матрицы Лямбда, которая используется при вычислении подынтегральной функции 
# для одной компоненты несимметризованного тензора a_not_sym в конкретном узле сетки интегрирования
@numba.njit()
def LYAMBDA_inversed(Cklmn, tetta, phi, a1, a2, a3):
    
    result = np.zeros((3,3))

    n_all = np.zeros(3)
    n_all[0] = np.sin(tetta) * np.cos(phi) / a1
    n_all[1] = np.sin(tetta) * np.sin(phi) / a2
    n_all[2] = np.cos(tetta) / a3

    for k in range(3):
        for l in range(3):
            for m in range(3):
                for n in range(3):
                    result[k,l] += Cklmn[k,m,l,n] * calculate_Nmn(tetta, phi, a1, a2 ,a3, n, m)
    
    return np.linalg.inv(result)

# Расчёт всех инвертированных матриц Лямбда для каждого узла сетки интегрирования
@numba.njit()
def get_all_inversed_LYAMBDA(tetha_s, phi_s, Cklmn, a1, a2, a3):

    LYAMBDA_inversed_all = np.zeros((len(tetha_s)*len(phi_s), 3, 3), dtype=float)

    counter = 0
    for i in tetha_s:
        for j in phi_s:
            LYAMBDA_inversed_all[counter] = LYAMBDA_inversed(Cklmn, i, j, a1, a2, a3)
            counter += 1

    return LYAMBDA_inversed_all

all_inversed_LYAMBDA_matrix = get_all_inversed_LYAMBDA(tetha_matrix, phi_matrix, Cklmn, a1_matrix, a2_matrix, a3_matrix)
all_inversed_LYAMBDA_fluid = get_all_inversed_LYAMBDA(tetha_fluid, phi_fluid, Cklmn, a1_fluid, a2_fluid, a3_fluid)

# Расчёт подынтегральной функции для одной компоненты несимметризованного тензора a_not_sym
# (задействуются все узлы сетки интегрирования)
@numba.njit()
def integrand_function_for_single_set_klmn(k, m, l, n, tetha_s, phi_s, all_inversed_LYAMBDA,a1, a2, a3):
    
    result = np.zeros((len(tetha_s), len(phi_s)))

    counter = 0
    for c_i, i in enumerate(tetha_s):
        for c_j, j in enumerate(phi_s):
            result[c_i, c_j] = calculate_Nmn(i,j,a1,a2,a3,n,m)*all_inversed_LYAMBDA[counter, k, l]*np.sin(i)
            counter += 1
    
    return result

# Расчёт подынтегральной функции для всех компонент несимметризованного тензора a_not_sym
@numba.njit()
def integrand_function_for_all_klmn(tetha_s, phi_s, all_inversed_LYAMBDA, a1, a2, a3):
    
    result = np.zeros((81, len(tetha_s), len(phi_s)), dtype=float)
    
    counter = 0
    for k in range(3):
        for m in range(3):
            for l in range(3):
                for n in range(3):
                    result[counter] = integrand_function_for_single_set_klmn(k,m,l,n,tetha_s,phi_s,all_inversed_LYAMBDA,a1,a2,a3)
                    counter +=1
    
    return result

a_all_integrand_function_matrix = integrand_function_for_all_klmn(tetha_matrix, phi_matrix, all_inversed_LYAMBDA_matrix, a1_matrix, a2_matrix, a3_matrix)
a_all_integrand_function_fluid = integrand_function_for_all_klmn(tetha_fluid, phi_fluid, all_inversed_LYAMBDA_fluid, a1_fluid, a2_fluid, a3_fluid)                   

# Отрисовка поверхностей для всех 81 подынтегральной функции. Сохранение в
# HTML-файлах для интерактивного изучения в браузере
def draw_integrand_function_plotly_for_isotripic_klmn(tetha_s, phi_s, a1, a2, a3, a):

    print(f"Построение для --изотропного случая-- (81) подынтегральных функций компонент тензора a_klmn")

    counter = 0
    for k in range(3):
        for m in range(3):
            for l in range(3):
                for n in range(3):
                    plotly.offline.plot({"data": go.Surface(z=a[counter],x=phi_s, y=tetha_s),
                                        "layout": go.Layout(title=f'Isotropic matrix {counter}/81   k={k+1}, m={m+1}, l={l+1}, n={n+1}, a1={a1}, a2={a2}, a3={a3}',
                                                scene = dict(xaxis_title_text='Phi', yaxis_title_text='Tetha',
                                                zaxis_title_text='Integrand_function'))},
                                                filename=f"matrix_integrand_function_{k+1}{m+1}{l+1}{n+1}.html")
                    counter +=1
                    print(f'{counter+1} / 81')

# draw_integrand_function_plotly_for_isotripic_klmn(tetha_matrix, phi_matrix, a1_matrix, a2_matrix, a3_matrix, a_all_integrand_function_matrix)

# Поиск ближайшего значения к заданному из списка для обработки
# пользовательского ввода при построении двумерных срезов
def take_closest(myList, myNumber):
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before

# Построение двумерных срезов для анализа подынтегральных функций
# в интерактивном режиме через консоль. Срезы сохраняются в формате HTML-файлов 
# для детального изучения в браузере
def slices_by_angeles(tetha_s, phi_s, a, a1, a2, a3, m_t):
    print("\n--- Построение срезов ---")
    while True:
        tetha_slices = []
        phi_slices = []
        i_f_num_user = []
        log_y_axes = False
        i_f_num_user = int(input("\nНомер подынтегральной функции: "))
        if input("\nЛогарифмический масштаб (yes/no): ") == "yes":
            log_y_axes = True
        else: log_y_axes = False
        for element in input("Набор углов tetha [0;Pi]: ").split():
            tetha_slices.append(float(element))
        for element in input("Набор углов phi [0;2*Pi]: ").split():
            phi_slices.append(float(element))
        for i,s in enumerate(tetha_slices):
            tetha_slices[i] = take_closest(tetha_s, s)
        for i,s in enumerate(phi_slices):
            phi_slices[i] = take_closest(phi_s, s)

        fig = make_subplots(rows=2, cols=1, subplot_titles=("Срезы по фиксированной tetha", "Срезы по фиксированному phi"))
        for i in tetha_slices:
            fig.append_trace(go.Scatter(
            x=phi_s,
            y=a[i_f_num_user-1][list(tetha_s).index(i),:],
            name="(tetha {0:1.2F})".format(i)), row=1, col=1)
        for i in phi_slices:
            fig.append_trace(go.Scatter(
            x=tetha_s,
            y=a[i_f_num_user-1][:,list(phi_s).index(i)],
            name="(phi {0:1.2F})".format(i)), row=2, col=1)
        
        fig.update_xaxes(title_text="Phi", row=1, col=1)
        fig.update_xaxes(title_text="Tetha", row=2, col=1)
        if log_y_axes ==True:
            fig.update_yaxes(title_text="Integrand_function", type="log", row=1, col=1)
            fig.update_yaxes(title_text="Integrand_function", type="log", row=2, col=1)
        else:
            fig.update_yaxes(title_text="Integrand_function", row=1, col=1)
            fig.update_yaxes(title_text="Integrand_function", row=2, col=1)

        fig.update_layout(title_text=f'Isotropic {m_t} {i_f_num_user}/81, a1={a1}, a2={a2}, a3={a3}')
        fig.write_html("slices.html")
        if input("\nПостроить новые срезы (yes/no): ") == "no":
            break
        else: continue

# slices_by_angeles(tetha_matrix, phi_matrix, a_all_integrand_function_fluid, a1_fluid, a2_fluid, a3_fluid, 'fluid')

# Численное интегрирование по поверхности для одной подынтегриальной функции.
# Вычиление одной компоненты несимметризованного тензора a_not_sym
@numba.njit()
def integral_calculation_by_method_of_medium_rectangles_for_single_klmn(tetha_s,phi_s,a):
    result = 0.0
    for x in range(a.shape[0]-1):
        for y in range(a.shape[1]-1):
            mean = (a[x][y] + a[x+1][y] + a[x][y+1] + a[x+1][y+1])/4
            step_x = tetha_s[x+1] - tetha_s[x]
            step_y = phi_s[y+1] - phi_s[y]
            result +=  mean*(step_x * step_y)
    result = -1/(4*np.pi)*result
    return result

# Численное интегрирование всех подынтегриальных функций.
# Вычиление всех компонент несимметризованного тензора a_not_sym
@numba.njit()
def integral_calculation_by_method_of_medium_rectangles_for_all(tetha_s, phi_s, a):
    
    #start_time = time.time()

    #print(f"Расчёт для --изотропного случая-- (81) компонент тензора a_klmn")

    result = np.zeros((3,3,3,3), dtype=float)

    counter = 0
    for k in range(3):
        for m in range(3):
            for l in range(3):
                for n in range(3):
                    result[k,m,l,n] = integral_calculation_by_method_of_medium_rectangles_for_single_klmn(tetha_s, phi_s, a[counter])
                    counter +=1
    
    # print("--- %s seconds ---" % (time.time() - start_time))
    # print(f"Для изотропного случая (81) компоненты тензора a_klmn занимают в памяти: {sys.getsizeof(result)/2**20} Mb\t")
    return result

a_klmn_all_matrix = integral_calculation_by_method_of_medium_rectangles_for_all(tetha_matrix, phi_matrix, a_all_integrand_function_matrix)
a_klmn_all_fluid = integral_calculation_by_method_of_medium_rectangles_for_all(tetha_fluid, phi_fluid, a_all_integrand_function_fluid)

# Симметризация рассчитанного тензора a_not_sym и вычисление
# тензора податливости g путём переприсваивания компонент тензора a_sym
@numba.njit()
def tensor_g_calculation_for_all_klmn(a):

    g = np.zeros((3,3,3,3), dtype=float)

    a_sym = np.zeros((3,3,3,3), dtype=float)
    
    # Симметризация
    for k in range(3):
        for l in range(3):
            for n in range(3):
                for m in range(3):
                    a_sym[k,l,n,m] = 0.25*(a[k,l,n,m]+a[m,l,n,k]+a[k,n,l,m]+a[m,n,l,k])

    # Переприсваивание
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    g[i,j,k,l] = a_sym[i,k,l,j]

    return g

g_matrix = tensor_g_calculation_for_all_klmn(a_klmn_all_matrix)
g_fluid = tensor_g_calculation_for_all_klmn(a_klmn_all_fluid)

# Расчёт инвертированного тензора 
@numba.njit()
def inverse(a, index):

    a = convert_full_stiffness_matrix_to_voigt(a, index)

    a = np.linalg.inv(a)

    a = convert_voigt_to_full_stiffness_matrix(a, index)

    return a

# Тензорное умножение (свёртка по внутренним индексам)
@numba.njit()
def multiplication(a, b):
    result = np.zeros((3,3,3,3), dtype=float)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    for m in range(3):
                        for n in range(3):
                            result[i][j][k][l] += a[i][j][m][n]*b[m][n][k][l]
    
    return result

# Итоговый расчёт эффективного тензора упругости. 
# В формуле вместо тензорных операций используется матричные за счёт 
# изначального приведения всех переменных в соответсвующему виду.
# При этом реализован функционал для расчёта в тензорном виде
def calculat_effective_elastic_properties(g_1, g_2, C_1, C_2, Cklmn, v_1, v_2):

    result = np.zeros((3,3,3,3), dtype=float)
    
    t_1 = np.zeros((3,3,3,3), dtype=float)
    t_2 = np.zeros((3,3,3,3), dtype=float)
    t_3 = np.zeros((3,3,3,3), dtype=float)
    t_4 = np.zeros((3,3,3,3), dtype=float)

    g_1 = convert_full_stiffness_matrix_to_voigt(g_1, True)
    g_2 = convert_full_stiffness_matrix_to_voigt(g_2, True)
    C_1 = convert_full_stiffness_matrix_to_voigt(C_1, False)
    C_2 = convert_full_stiffness_matrix_to_voigt(C_2, False)
    Cklmn = convert_full_stiffness_matrix_to_voigt(Cklmn, False)

    t_1_1 = np.linalg.inv(g_1)
    t_1_2 = np.linalg.inv(t_1_1-(C_1-Cklmn))
    t_1_3 = np.matmul(t_1_2, t_1_1)
    t_1_4 = np.matmul(C_1,t_1_3)
    t_1 = -v_1*t_1_4

    t_2_1 = np.linalg.inv(g_2)
    t_2_2 = np.linalg.inv(t_2_1-(C_2-Cklmn))
    t_2_3 = np.matmul(t_2_2, t_2_1)
    t_2_4 = np.matmul(C_2,t_2_3)
    t_2 = -v_2*t_2_4

    t_3_1 = np.linalg.inv(g_1)
    t_3_2 = np.linalg.inv(t_3_1-(C_1-Cklmn))
    t_3_3 = np.matmul(t_3_2, t_3_1)
    t_3 = -v_1*t_3_3

    t_4_1 = np.linalg.inv(g_2)
    t_4_2 = np.linalg.inv(t_4_1-(C_2-Cklmn))
    t_4_3 = np.matmul(t_4_2, t_4_1)
    t_4 = -v_2*t_4_3

    result = np.matmul((t_1+t_2),np.linalg.inv(t_3+t_4))
    
    return result

C = calculat_effective_elastic_properties(g_matrix,g_fluid,C_matrix,C_fluid,Cklmn,v_matrix,v_fluid)

print(np.round(C, 6))

# Benchmark предназначен для тестирования многократного расчёта по методу GSA для 
# проведения прямого сравнения с версиями кода на Rust и Fortran
def benchmark():

    num_rounds = 3

    bm_s_t = time.time()

    for i in range(num_rounds):
        k_matrix: float = random.uniform(30,40)
        mu_matrix: float = random.uniform(15,25)
        v_matrix: float =  random.uniform(0.85,0.999)
        a1_matrix: float = 1.0
        a2_matrix: float = 1.0
        a3_matrix: float = 1.0

        
        k_fluid: float = random.uniform(2,5)
        mu_fluid: float = 0.0
        v_fluid: float =  1 - v_matrix
        a1_fluid: float = 1000.0
        a2_fluid: float = 1000.0
        a3_fluid: float = 1.0

        f: float = random.uniform(0.7,0.9)

        C_matrix = calculate_Cklmn_from_k_mu(k_matrix, mu_matrix)
        C_fluid = calculate_Cklmn_from_k_mu(k_fluid, mu_fluid)
        Cklmn = (1-f)*C_matrix+f*C_fluid
        
        tetha_matrix, phi_matrix, tetha_fluid, phi_fluid = get_axes_for_all_areas(range_tetha_matrix, range_phi_matrix, range_tetha_fluid, range_phi_fluid)
        
        all_inversed_LYAMBDA_matrix = get_all_inversed_LYAMBDA(tetha_matrix, phi_matrix, Cklmn, a1_matrix, a2_matrix, a3_matrix)
        all_inversed_LYAMBDA_fluid = get_all_inversed_LYAMBDA(tetha_fluid, phi_fluid, Cklmn, a1_fluid, a2_fluid, a3_fluid)

        a_all_integrand_function_matrix = integrand_function_for_all_klmn(tetha_matrix, phi_matrix, all_inversed_LYAMBDA_matrix, a1_matrix, a2_matrix, a3_matrix)
        a_all_integrand_function_fluid = integrand_function_for_all_klmn(tetha_fluid, phi_fluid, all_inversed_LYAMBDA_fluid, a1_fluid, a2_fluid, a3_fluid)

        a_klmn_all_matrix = integral_calculation_by_method_of_medium_rectangles_for_all(tetha_matrix, phi_matrix, a_all_integrand_function_matrix)
        a_klmn_all_fluid = integral_calculation_by_method_of_medium_rectangles_for_all(tetha_fluid, phi_fluid, a_all_integrand_function_fluid)

        g_matrix = tensor_g_calculation_for_all_klmn(a_klmn_all_matrix)
        g_fluid = tensor_g_calculation_for_all_klmn(a_klmn_all_fluid)

        C = calculat_effective_elastic_properties(g_matrix,g_fluid,C_matrix,C_fluid,Cklmn,v_matrix,v_fluid)
    print("Benchmark time")
    print("--- %s seconds ---" % (time.time() - bm_s_t))
    print(f"Num rounds: {num_rounds}")
    print("\n")

# benchmark()

print("Programm time")
print("--- %s seconds ---" % (time.time() - s_t))

del all_inversed_LYAMBDA_matrix
del all_inversed_LYAMBDA_fluid
del a_all_integrand_function_matrix
del a_all_integrand_function_fluid
del a_klmn_all_matrix
del a_klmn_all_fluid
del g_matrix
del g_fluid
del C
