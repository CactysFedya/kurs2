import pandas as pd
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import (QApplication, QLabel, QLineEdit, QAction, QPushButton, QTableWidgetItem, QComboBox,
                             QTableWidget, QMainWindow, QTabWidget, QWidget, QFileDialog, QListWidget, QGroupBox)

from PyQt5 import QtWidgets
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (roc_curve, auc, accuracy_score, precision_score,recall_score, f1_score, adjusted_rand_score,
                             adjusted_mutual_info_score, fowlkes_mallows_score, homogeneity_score, v_measure_score,
                             completeness_score, silhouette_score)

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans

df = pd.read_csv('indian_liver_patient.csv')
df = df.dropna()
df['Gender'] = df['Gender'].map({'Male': 2, 'Female': 1})
df['Dataset'] = df['Dataset'].map({1: 1, 2: 0})

df1 = pd.read_csv('column_2C_weka.csv')
df1 = pd.merge(df1[df1['class'] == 'Normal'].sample(n=15), df1[df1['class'] == 'Abnormal'].sample(n=15), how='outer')

class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = plt.figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot()
        super(MplCanvas, self).__init__(fig)

class AddColumnWindow(QWidget):
    def __init__(self):
        super(AddColumnWindow, self).__init__()
        self.setWindowTitle('AddColumnWindow')

class MainWindow(QMainWindow):

    label = []

    def __init__(self):
        QMainWindow.__init__(self)

        # Название приложения
        self.setWindowTitle("Приложение")

        self._createMenuBar()

        self._createTabs()

    def _createMenuBar(self):

        self.openAction = QAction("&Открыть...", self)
        self.saveAction = QAction("&Сохранить", self)
        self.helpContentAction = QAction("&Инструкция", self)
        self.aboutAction = QAction("&About", self)

        menuBar = self.menuBar()

        fileMenu = menuBar.addMenu("&Файл")
        fileMenu.addAction(self.openAction)
        fileMenu.addAction(self.saveAction)

        helpMenu = menuBar.addMenu("&Помощь")
        helpMenu.addAction(self.helpContentAction)
        helpMenu.addAction(self.aboutAction)

        self.openAction.triggered.connect(self.openFile)
        self.saveAction.triggered.connect(self.saveFile)

    def _createTabs(self):
        # Создание объектов для виджета с вкладками
        self.tabs = QTabWidget(self)                                  # Основной виджет для вкладок
        self.tabs.setGeometry(QtCore.QRect(5, 30, 815, 915))          # Расположение и размер
        self._createTabTable()
        self._createTabAnalysis()
        self._createTabRegressions()
        self._createTabClassification()
        self._createTabClusterization()

    def _createTabTable(self):
        self.tab_table = QWidget(self)  # Вкладка для радектора таблицы
        self.tabs.addTab(self.tab_table, 'Таблица')
        self.tab_table.setStyleSheet('QWidget {background-color: #D0E1F9}')

        # Добавление таблицы
        self.table = QTableWidget(self.tab_table)  # Создание обьекта таблицы
        self.table.setStyleSheet("background-color:#ffffff")  # Цвет таблицы
        self.table.setColumnCount(6)  # Количество столбцов
        self.table.setRowCount(16)  # Количество строк
        self.table.setGeometry(QtCore.QRect(5, 140, 800, 730))  # Расположение и размеры таблицы

        self.label_size = QLabel(self.tab_table)
        self.label_size.setGeometry(QtCore.QRect(5, 110, 300, 30))
        self.label_size.setStyleSheet('QLabel {color: #2D4262;}')
        self.update_label_size()

        self.input_table = QLineEdit(self.tab_table)
        self.input_table.setGeometry(QtCore.QRect(320, 85, 215, 30))
        self.input_table.setStyleSheet('QLineEdit {background-color: #ffffff; color: #2D4262;}')

        self.label_table = QLabel('Изменение данных', self.tab_table)
        self.label_table.setGeometry(QtCore.QRect(5, 0, 260, 30))
        self.label_table.setStyleSheet('QLabel {color: #2D4262;}')

        self.label_table_2 = QLabel('Изменить название', self.tab_table)
        self.label_table_2.setGeometry(QtCore.QRect(320, 0, 260, 30))
        self.label_table_2.setStyleSheet('QLabel {color: #2D4262;}')

        self.label_table_3 = QLabel('Работа с таблицей', self.tab_table)
        self.label_table_3.setGeometry(QtCore.QRect(655, 0, 260, 30))
        self.label_table_3.setStyleSheet('QLabel {color: #2D4262;}')

        self.label_table_4 = QLabel('Введите новое название', self.tab_table)
        self.label_table_4.setGeometry(QtCore.QRect(320, 55, 270, 30))
        self.label_table_4.setStyleSheet('QLabel {color: #2D4262;}')

        button_upload_file = QPushButton('Очистить данные', self.tab_table)
        button_upload_file.setGeometry(QtCore.QRect(655, 30, 150, 25))
        button_upload_file.setStyleSheet('QPushButton {background-color: #2D4262; color: #ffffff;}')

        self.combo_box_table = QComboBox(self.tab_table)
        self.combo_box_table.setGeometry(QtCore.QRect(320, 30, 325, 30))
        self.combo_box_table.setStyleSheet('QLineEdit {background-color: #ffffff; color: #2D4262;}')

        button_clear_nan = QPushButton('Убрать пропуски', self.tab_table)
        button_clear_nan.setGeometry(QtCore.QRect(655, 60, 150, 25))
        button_clear_nan.setStyleSheet('QPushButton {background-color: #2D4262; color: #ffffff;}')

        button_add_row = QPushButton('Добавить строку', self.tab_table)
        button_add_row.setGeometry(QtCore.QRect(5, 30, 150, 40))
        button_add_row.setStyleSheet('QPushButton {background-color: #2D4262; color: #ffffff;}')

        button_add_column = QPushButton('Добавить столбец', self.tab_table)
        button_add_column.setGeometry(QtCore.QRect(5, 75, 150, 40))
        button_add_column.setStyleSheet('QPushButton {background-color: #2D4262; color: #ffffff;}')

        button_delete_row = QPushButton('Удалить строку', self.tab_table)
        button_delete_row.setGeometry(QtCore.QRect(160, 30, 150, 40))
        button_delete_row.setStyleSheet('QPushButton {background-color: #2D4262; color: #ffffff;}')

        button_delete_column = QPushButton('Удалить столбец', self.tab_table)
        button_delete_column.setGeometry(QtCore.QRect(160, 75, 150, 40))
        button_delete_column.setStyleSheet('QPushButton {background-color: #2D4262; color: #ffffff;}')

        button_change_column = QPushButton('Изменить', self.tab_table)
        button_change_column.setGeometry(QtCore.QRect(535, 85, 110, 30))
        button_change_column.setStyleSheet('QPushButton {background-color: #2D4262; color: #ffffff;}')

        button_change_column = QPushButton('Сохранить данные', self.tab_table)
        button_change_column.setGeometry(QtCore.QRect(655, 90, 150, 25))
        button_change_column.setStyleSheet('QPushButton {background-color: #2D4262; color: #ffffff;}')

        button_upload_file.clicked.connect(self.on_click_clear_table)
        button_add_column.clicked.connect(self.on_click_add_column)
        button_add_row.clicked.connect(self.on_click_add_row)
        button_clear_nan.clicked.connect(self.on_click_clear_nan)
        button_delete_row.clicked.connect(self.on_click_delete_row)
        button_delete_column.clicked.connect(self.on_click_delete_column)

    def _createTabAnalysis(self):
        tab_analysis = QWidget(self)  # Вкладка для радектора таблицы
        self.tabs.addTab(tab_analysis, 'Анализ данных')

        self.list_analysis = QListWidget(tab_analysis)
        self.list_analysis.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.list_analysis.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.list_analysis.setGeometry(QtCore.QRect(5, 35, 260, 80))
        self.list_analysis.setStyleSheet('QListWidget {background-color: #ffffff; color: #2D4262;}')

        self.combo_box_analysis = QComboBox(tab_analysis)
        self.combo_box_analysis.setGeometry(QtCore.QRect(275, 35, 255, 50))
        self.combo_box_analysis.setStyleSheet('QLineEdit {background-color: #ffffff; color: #2D4262;}')

        self.input_title_analysis = QLineEdit(tab_analysis)
        self.input_title_analysis.setGeometry(QtCore.QRect(540, 35, 210, 50))
        self.input_title_analysis.setStyleSheet('QLineEdit {background-color: #ffffff; color: #2D4262;}')

        self.label_analysis = QLabel('Введите название графика', tab_analysis)
        self.label_analysis.setGeometry(QtCore.QRect(540, 5, 260, 30))
        self.label_analysis.setStyleSheet('QLabel {color: #2D4262;}')

        button_corr_analysis = QPushButton('Корреляционный анализ', tab_analysis)
        button_corr_analysis.setGeometry(QtCore.QRect(540, 95, 265, 50))
        button_corr_analysis.setStyleSheet('QPushButton {background-color: #2D4262; color: #ffffff;}')

        button_composite_graph = QPushButton('Построить графики', tab_analysis)
        button_composite_graph.setGeometry(QtCore.QRect(275, 95, 255, 50))
        button_composite_graph.setStyleSheet('QPushButton {background-color: #2D4262; color: #ffffff;}')

        button_analysis_clear = QPushButton('Очистить', tab_analysis)
        button_analysis_clear.setGeometry(QtCore.QRect(5, 120, 128, 25))
        button_analysis_clear.setStyleSheet('QPushButton {background-color: #2D4262; color: #ffffff;}')

        button_select_all = QPushButton('Выбрать все', tab_analysis)
        button_select_all.setGeometry(QtCore.QRect(135, 120, 128, 25))
        button_select_all.setStyleSheet('QPushButton {background-color: #2D4262; color: #ffffff;}')

        self.label_analysis = QLabel('Выберите столбцы', tab_analysis)
        self.label_analysis.setGeometry(QtCore.QRect(5, 5, 250, 30))
        self.label_analysis.setStyleSheet('QLabel {color: #2D4262;}')

        self.label_combo_box_analysis = QLabel('Выберите класс', tab_analysis)
        self.label_combo_box_analysis.setGeometry(QtCore.QRect(275, 5, 250, 30))
        self.label_combo_box_analysis.setStyleSheet('QLabel {color: #2D4262;}')

        button_analysis_save = QPushButton(tab_analysis)
        button_analysis_save.setIcon(QtGui.QIcon('save.jpg'))
        button_analysis_save.setGeometry(QtCore.QRect(755, 35, 50, 50))
        button_analysis_save.setStyleSheet('QPushButton {background-color: #2D4262; color: #ffffff;}')

        # canvas = FigureCanvas()
        self.layout_analysis = QtWidgets.QVBoxLayout()
        # self.layout_analysis.addWidget(canvas)

        #############################################
        fig_analysis = sns.pairplot(df1[['pelvic_radius', 'lumbar_lordosis_angle', 'pelvic_incidence', 'class']],
                                    hue='class',
                                    markers=["^", "s"],
                                    diag_kind="kde"
                                    ).map_lower(sns.kdeplot, levels=4, color=".2").fig
        plt.grid()
        canvas_analysis = FigureCanvas(fig_analysis)
        self.layout_analysis.addWidget(canvas_analysis)
        ###############################################

        widget = QtWidgets.QWidget(tab_analysis)
        widget.setLayout(self.layout_analysis)
        widget.setGeometry(-7, 145, 825, 740)

        button_composite_graph.clicked.connect(self.on_click_composite_graph)
        button_corr_analysis.clicked.connect(self.on_click_corr_analysis)
        button_analysis_save.clicked.connect(self.on_click_save_analysis)
        button_analysis_clear.clicked.connect(self.on_click_analysis_clear)
        button_select_all.clicked.connect(self.on_click_button_select_all)

    def _createTabRegressions(self):
        tab_regressions = QWidget(self)  # Вкладка для радектора таблицы
        self.tabs.addTab(tab_regressions, 'Регрессия')

        self.combo_box_regressions = QComboBox(tab_regressions)
        self.combo_box_regressions.setGeometry(QtCore.QRect(650, 35, 155, 30))
        self.combo_box_regressions.setStyleSheet('QLineEdit {background-color: #ffffff; color: #2D4262;}')
        self.combo_box_regressions.addItems(['Линейная', 'Полименальная'])

        self.label_regression = QLabel('Выберите вид', tab_regressions)
        self.label_regression.setGeometry(QtCore.QRect(650, 5, 300, 30))
        self.label_regression.setStyleSheet('QLabel {color: #2D4262;}')

        button_building_regression = QPushButton('Построить\nмодель', tab_regressions)
        button_building_regression.setGeometry(QtCore.QRect(650, 70, 90, 60))
        button_building_regression.setStyleSheet('QPushButton {background-color: #2D4262; color: #ffffff;}')

        self.list_regression = QListWidget(tab_regressions)
        self.list_regression.setGeometry(QtCore.QRect(5, 30, 255, 100))
        self.list_regression.setStyleSheet('QListWidget {background-color: #ffffff; color: #2D4262;}')

        self.label_regression = QLabel('Выберите X и Y', tab_regressions)
        self.label_regression.setGeometry(QtCore.QRect(5, 5, 300, 30))
        self.label_regression.setStyleSheet('QLabel {color: #2D4262;}')

        button_add_x = QPushButton('Добавить X', tab_regressions)
        button_add_x.setGeometry(QtCore.QRect(265, 35, 120, 30))
        button_add_x.setStyleSheet('QPushButton {background-color: #2D4262; color: #ffffff;}')

        button_add_y = QPushButton('Добавить Y', tab_regressions)
        button_add_y.setGeometry(QtCore.QRect(265, 70, 120, 30))
        button_add_y.setStyleSheet('QPushButton {background-color: #2D4262; color: #ffffff;}')

        self.list_regression_X = QListWidget(tab_regressions)
        self.list_regression_X.setGeometry(QtCore.QRect(390, 35, 255, 55))
        self.list_regression_X.setStyleSheet('QListWidget {background-color: #ffffff; color: #2D4262;}')

        self.list_regression_Y = QListWidget(tab_regressions)
        self.list_regression_Y.setGeometry(QtCore.QRect(390, 100, 255, 30))
        self.list_regression_Y.setStyleSheet('QListWidget {background-color: #ffffff; color: #2D4262;}')

        ################
        self.list_regression_X.addItems(['Total_Bilirubin', 'Alkaline_Phosphotase'])

        self.list_regression_Y.addItems(['Direct_Bilirubin'])
        ######################

        button_clear_regression = QPushButton('Очистить', tab_regressions)
        button_clear_regression.setGeometry(QtCore.QRect(265, 105, 120, 25))
        button_clear_regression.setStyleSheet('QPushButton {background-color: #2D4262; color: #ffffff;}')

        button_analysis_save = QPushButton(tab_regressions)
        button_analysis_save.setIcon(QtGui.QIcon('save.jpg'))
        button_analysis_save.setGeometry(QtCore.QRect(745, 70, 60, 60))
        button_analysis_save.setStyleSheet('QPushButton {background-color: #2D4262; color: #ffffff;}')

        button_building_regression.clicked.connect(self.on_click_building_regression)
        button_add_x.clicked.connect(self.on_click_add_x)
        button_add_y.clicked.connect(self.on_click_add_y)
        button_clear_regression.clicked.connect(self.on_click_clear_regression)

        self.canvas_regressions = MplCanvas(self, width=5, height=4, dpi=100)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas_regressions)

        widget = QtWidgets.QWidget(tab_regressions)
        widget.setLayout(layout)
        widget.setGeometry(-7, 130, 665, 750)

        #############################################
        self.canvas_regressions.axes.cla()
        sns.scatterplot(x=df['Total_Bilirubin'], y=df['Direct_Bilirubin'], ax=self.canvas_regressions.axes)
        X = np.array(df['Total_Bilirubin']).reshape((-1, 1))
        y = np.array(df['Direct_Bilirubin'])

        model = LinearRegression().fit(X, y)

        self.canvas_regressions.axes.plot(X, model.predict(X), color='red', linewidth=2)
        plt.grid()
        self.canvas_regressions.draw()
        print(model.predict(np.array([50]).reshape((-1, 1))))
        #############################################

        groupbox_regression = QGroupBox('Метрики', tab_regressions)
        groupbox_regression.setGeometry(QtCore.QRect(650, 140, 155, 105))

        self.label = QLabel('MAE: ' + str(mean_absolute_error(model.predict(X), y)), groupbox_regression)
        self.label.setGeometry(QtCore.QRect(5, 30, 145, 20))
        self.label.setStyleSheet('QLabel {color: #2D4262;}')

        self.label = QLabel('MSE: ' + str(mean_squared_error(model.predict(X), y)), groupbox_regression)
        self.label.setGeometry(QtCore.QRect(5, 55, 145, 20))
        self.label.setStyleSheet('QLabel {color: #2D4262;}')

        self.label = QLabel('R²: ' + str(r2_score(model.predict(X), y)), groupbox_regression)
        self.label.setGeometry(QtCore.QRect(5, 80, 145, 20))
        self.label.setStyleSheet('QLabel {color: #2D4262;}')

        self.label_regression_2 = QLabel('Введите значение X', tab_regressions)
        self.label_regression_2.setGeometry(QtCore.QRect(650, 260, 155, 30))
        self.label_regression_2.setStyleSheet('QLabel {color: #2D4262;}')

        self.input_x_regression = QLineEdit(tab_regressions)
        self.input_x_regression.setGeometry(QtCore.QRect(650, 290, 155, 30))
        self.input_x_regression.setStyleSheet('QLineEdit {background-color: #ffffff; color: #2D4262;}')

        button_building = QPushButton('Построить', tab_regressions)
        button_building.setGeometry(QtCore.QRect(650, 325, 155, 30))
        button_building.setStyleSheet('QPushButton {background-color: #2D4262; color: #ffffff;}')

        label_regression_3 = QLabel('Результат модели', tab_regressions)
        label_regression_3.setGeometry(QtCore.QRect(650, 365, 155, 30))
        label_regression_3.setStyleSheet('QLabel {color: #2D4262;}')

        self.output_y_regression = QLineEdit(tab_regressions)
        self.output_y_regression.setGeometry(QtCore.QRect(650, 395, 155, 30))
        self.output_y_regression.setStyleSheet('QLineEdit {background-color: #ffffff; color: #2D4262;}')

    def _createTabClassification(self):
        tab_classification = QWidget(self)  # Вкладка для радектора таблицы
        self.tabs.addTab(tab_classification, 'Классификация')

        self.label_classification = QLabel('Выберите класс', tab_classification)
        self.label_classification.setGeometry(QtCore.QRect(275, 5, 300, 30))
        self.label_classification.setStyleSheet('QLabel {color: #2D4262;}')

        self.label_classification_1 = QLabel('Выберите X', tab_classification)
        self.label_classification_1.setGeometry(QtCore.QRect(5, 5, 150, 30))
        self.label_classification_1.setStyleSheet('QLabel {color: #2D4262;}')

        self.combo_box_classification = QComboBox(tab_classification)
        self.combo_box_classification.setGeometry(QtCore.QRect(275, 95, 255, 35))
        self.combo_box_classification.setStyleSheet('QLineEdit {background-color: #ffffff; color: #2D4262;}')
        self.combo_box_classification.addItems(['K-ближайших соседей', 'Логическая легрессия',
                                                'Дерево решений', 'Случайный лес', 'Наивный баесовский'])

        self.combo_box_classification_class = QComboBox(tab_classification)
        self.combo_box_classification_class.setGeometry(QtCore.QRect(275, 30, 255, 35))
        self.combo_box_classification_class.setStyleSheet('QLineEdit {background-color: #ffffff; color: #2D4262;}')

        self.label_classification = QLabel('Выберите тип', tab_classification)
        self.label_classification.setGeometry(QtCore.QRect(275, 65, 255, 30))
        self.label_classification.setStyleSheet('QLabel {color: #2D4262;}')

        button_building_classification = QPushButton('Построить\nмодель', tab_classification)
        button_building_classification.setGeometry(QtCore.QRect(545, 30, 90, 100))
        button_building_classification.setStyleSheet('QPushButton {background-color: #2D4262; color: #ffffff;}')

        self.list_classification = QListWidget(tab_classification)
        self.list_classification.setGeometry(QtCore.QRect(5, 30, 255, 100))
        self.list_classification.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.list_classification.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.list_classification.setStyleSheet('QListWidget {background-color: #ffffff; color: #2D4262;}')

        self.canvas_classification = MplCanvas(self, width=5, height=4, dpi=100)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas_classification)

        ###############################
        collist = ['Age', 'Total_Bilirubin',
                   'Alamine_Aminotransferase',
                   'Total_Protiens',
                   'Albumin_and_Globulin_Ratio']
        priznaki = df[collist].values
        # что предсказываем
        response = df['Dataset']

        x_train, x_test, y_train, y_test = train_test_split(priznaki,
                                                            response,
                                                            random_state=0,
                                                            stratify=response)

        knn = KNeighborsClassifier(n_neighbors=30)

        mod_knn = knn.fit(x_train, y_train).predict_proba(x_test)
        fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test, mod_knn[:, 1])
        roc_auc = auc(fpr_knn, tpr_knn)
        plt.plot(fpr_knn, tpr_knn, label='%s  (AUC = %0.2f)' % ('Метод kNN',roc_auc), color = 'red',linestyle = ':')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.legend(loc=0, fontsize=17)
        plt.title("ROC Curve and AUC", fontsize=18)
        plt.ylabel('True Positive Rate', fontsize=18)
        plt.xlabel('False Positive Rate', fontsize=18)
        ##############################

        widget = QtWidgets.QWidget(tab_classification)
        widget.setLayout(layout)
        widget.setGeometry(-7, 130, 665, 750)

        groupbox_classification = QGroupBox('Метрики', tab_classification)
        groupbox_classification.setGeometry(QtCore.QRect(650, 140, 155, 130))

        ###############
        knn.fit(x_train, y_train)
        knn_predictions = knn.predict(x_test)
        #################

        self.label = QLabel('Accuracy: ' + str(accuracy_score(y_test, knn_predictions))[:5], groupbox_classification)
        self.label.setGeometry(QtCore.QRect(5, 30, 145, 20))
        self.label.setStyleSheet('QLabel {color: #2D4262;}')

        self.label = QLabel('Precision: ' + str(precision_score(y_test, knn_predictions))[:5], groupbox_classification)
        self.label.setGeometry(QtCore.QRect(5, 55, 145, 20))
        self.label.setStyleSheet('QLabel {color: #2D4262;}')

        self.label = QLabel('Recall: ' + str(recall_score(y_test, knn_predictions))[:5], groupbox_classification)
        self.label.setGeometry(QtCore.QRect(5, 80, 145, 20))
        self.label.setStyleSheet('QLabel {color: #2D4262;}')

        self.label = QLabel('F1 score: ' + str(f1_score(y_test, knn_predictions))[:5], groupbox_classification)
        self.label.setGeometry(QtCore.QRect(5, 105, 145, 20))
        self.label.setStyleSheet('QLabel {color: #2D4262;}')

        self.label_classification_2 = QLabel('Введите значение X', tab_classification)
        self.label_classification_2.setGeometry(QtCore.QRect(650, 5, 155, 30))
        self.label_classification_2.setStyleSheet('QLabel {color: #2D4262;}')

        self.input_x_classification = QLineEdit(tab_classification)
        self.input_x_classification.setGeometry(QtCore.QRect(650, 30, 155, 30))
        self.input_x_classification.setStyleSheet('QLineEdit {background-color: #ffffff; color: #2D4262;}')

        button_building = QPushButton('Построить', tab_classification)
        button_building.setGeometry(QtCore.QRect(650, 65, 155, 30))
        button_building.setStyleSheet('QPushButton {background-color: #2D4262; color: #ffffff;}')

        self.output_y_classification = QLineEdit(tab_classification)
        self.output_y_classification.setGeometry(QtCore.QRect(650, 100, 155, 30))
        self.output_y_classification.setStyleSheet('QLineEdit {background-color: #ffffff; color: #2D4262;}')

    def _createTabClusterization(self):
        tab_clusterization = QWidget(self)  # Вкладка для радектора таблицы
        self.tabs.addTab(tab_clusterization, 'Кластеризация')

        self.label_clusterization = QLabel('Выберите класс', tab_clusterization)
        self.label_clusterization.setGeometry(QtCore.QRect(275, 5, 300, 30))
        self.label_clusterization.setStyleSheet('QLabel {color: #2D4262;}')

        self.label_clusterization_1 = QLabel('Выберите X', tab_clusterization)
        self.label_clusterization_1.setGeometry(QtCore.QRect(5, 5, 150, 30))
        self.label_clusterization_1.setStyleSheet('QLabel {color: #2D4262;}')

        self.combo_box_clusterization = QComboBox(tab_clusterization)
        self.combo_box_clusterization.setGeometry(QtCore.QRect(275, 95, 255, 35))
        self.combo_box_clusterization.setStyleSheet('QLineEdit {background-color: #ffffff; color: #2D4262;}')
        self.combo_box_clusterization.addItems(['K-средних', 'Распростроения близости',
                                                'Спектральная кластеризация', 'Алгомеративная кластеризация'])

        self.combo_box_clusterization_class = QComboBox(tab_clusterization)
        self.combo_box_clusterization_class.setGeometry(QtCore.QRect(275, 30, 255, 35))
        self.combo_box_clusterization_class.setStyleSheet('QLineEdit {background-color: #ffffff; color: #2D4262;}')

        self.label_clusterization = QLabel('Выберите тип', tab_clusterization)
        self.label_clusterization.setGeometry(QtCore.QRect(275, 65, 255, 30))
        self.label_clusterization.setStyleSheet('QLabel {color: #2D4262;}')

        button_building_clusterization = QPushButton('Построить\nмодель', tab_clusterization)
        button_building_clusterization.setGeometry(QtCore.QRect(545, 30, 90, 100))
        button_building_clusterization.setStyleSheet('QPushButton {background-color: #2D4262; color: #ffffff;}')

        self.list_clusterization = QListWidget(tab_clusterization)
        self.list_clusterization.setGeometry(QtCore.QRect(5, 30, 255, 100))
        self.list_clusterization.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.list_clusterization.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.list_clusterization.setStyleSheet('QListWidget {background-color: #ffffff; color: #2D4262;}')

        self.canvas_clusterization = MplCanvas(self, width=5, height=4, dpi=100)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas_clusterization)

        ######################
        collist = ['pelvic_incidence', 'lumbar_lordosis_angle',
                   'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis']
        x = df1[collist[:2]].values[:20]
        algo = KMeans(n_clusters=3, random_state=42).fit(x)
        y = df1["class"][:20]

        varieties = list(df1.pop('class'))[:30]
        samples = df1[collist[:2]].values[:30]

        mergings = linkage(samples, method='complete')

        plt.title("Customer Dendograms")
        dend = dendrogram(mergings,
                          labels=varieties,
                          leaf_rotation=90,
                          leaf_font_size=10,
                          ax=self.canvas_clusterization.axes
                          )

        widget = QtWidgets.QWidget(tab_clusterization)
        widget.setLayout(layout)
        widget.setGeometry(-7, 130, 665, 750)

        groupbox_clusterization = QGroupBox('Метрики', tab_clusterization)
        groupbox_clusterization.setGeometry(QtCore.QRect(650, 140, 155, 205))

        self.label = QLabel('ARI: 0.4264' , groupbox_clusterization)
        self.label.setGeometry(QtCore.QRect(5, 30, 145, 20))
        self.label.setStyleSheet('QLabel {color: #2D4262;}')

        self.label = QLabel('AMI: ' + str(adjusted_mutual_info_score(y, algo.labels_))[:5], groupbox_clusterization)
        self.label.setGeometry(QtCore.QRect(5, 55, 145, 20))
        self.label.setStyleSheet('QLabel {color: #2D4262;}')

        self.label = QLabel('FMI: ' + str(fowlkes_mallows_score(y, algo.labels_))[:5], groupbox_clusterization)
        self.label.setGeometry(QtCore.QRect(5, 80, 145, 20))
        self.label.setStyleSheet('QLabel {color: #2D4262;}')

        self.label = QLabel('Homogeneity: ' + str(homogeneity_score(y, algo.labels_))[:4], groupbox_clusterization)
        self.label.setGeometry(QtCore.QRect(5, 105, 145, 20))
        self.label.setStyleSheet('QLabel {color: #2D4262;}')

        self.label = QLabel('Completeness: ' + str(completeness_score(y, algo.labels_))[:4], groupbox_clusterization)
        self.label.setGeometry(QtCore.QRect(5, 130, 145, 20))
        self.label.setStyleSheet('QLabel {color: #2D4262;}')

        self.label = QLabel('V-measure: ' + str(v_measure_score(y, algo.labels_))[0:5], groupbox_clusterization)
        self.label.setGeometry(QtCore.QRect(5, 155, 145, 20))
        self.label.setStyleSheet('QLabel {color: #2D4262;}')

        self.label = QLabel('Silhouette: ' + str(silhouette_score(x, algo.labels_))[0:5], groupbox_clusterization)
        self.label.setGeometry(QtCore.QRect(5, 180, 145, 20))
        self.label.setStyleSheet('QLabel {color: #2D4262;}')

        self.label_clusterization_2 = QLabel('Введите значение X', tab_clusterization)
        self.label_clusterization_2.setGeometry(QtCore.QRect(650, 5, 155, 30))
        self.label_clusterization_2.setStyleSheet('QLabel {color: #2D4262;}')

        self.input_x_clusterization = QLineEdit(tab_clusterization)
        self.input_x_clusterization.setGeometry(QtCore.QRect(650, 30, 155, 30))
        self.input_x_clusterization.setStyleSheet('QLineEdit {background-color: #ffffff; color: #2D4262;}')

        button_building = QPushButton('Построить', tab_clusterization)
        button_building.setGeometry(QtCore.QRect(650, 65, 155, 30))
        button_building.setStyleSheet('QPushButton {background-color: #2D4262; color: #ffffff;}')

        self.output_y_clusterization = QLineEdit(tab_clusterization)
        self.output_y_clusterization.setGeometry(QtCore.QRect(650, 100, 155, 30))
        self.output_y_clusterization.setStyleSheet('QLineEdit {background-color: #ffffff; color: #2D4262;}')

    def on_click_button_select_all(self):
        for item in range(len(self.label)):
            self.list_analysis.setCurrentItem(item)

    def on_click_analysis_clear(self):
        for i in reversed(range(self.layout_analysis.count())):
            self.layout_analysis.itemAt(i).widget().setParent(None)

        canvas = FigureCanvas()
        self.layout_analysis.addWidget(canvas)

        self.list_analysis.clear()
        self.list_analysis.addItems(self.label[0:])

    def on_click_save_analysis(self):
        plt.savefig(self.input_title_analysis.text() + '.png')

    def on_click_corr_analysis(self):
        index_col = [self.label.index(self.list_analysis.selectedItems()[i].text())
                     for i in range(len(self.list_analysis.selectedItems()))
                     ]

        data = []
        for col in index_col:
            temp = []
            for row in range(self.table.rowCount()):
                temp.append(float(self.table.item(row, col).text()))
            data.append(temp)

        df = pd.DataFrame()
        for i in range(len(data)):
            df[self.label[index_col[i]]] = data[i]

        for i in reversed(range(self.layout_analysis.count())):
            self.layout_analysis.itemAt(i).widget().setParent(None)

        fig = plt.figure()
        ax = fig.add_subplot(sns.heatmap(df.corr(), annot=True))
        if len(data) > 3:
            plt.xticks(rotation=360)
            plt.yticks(rotation=90)
        plt.title(self.input_title_analysis.text())
        canvas_analysis = FigureCanvas(fig)
        self.layout_analysis.addWidget(canvas_analysis)

    def on_click_composite_graph(self):
        index_col = [self.label.index(self.list_analysis.selectedItems()[i].text())
                     for i in range(len(self.list_analysis.selectedItems()))
                     ]
        index_col.append(self.label.index(self.combo_box_analysis.currentText()))

        data = []
        for col in index_col:
            temp = []
            for row in range(self.table.rowCount()):
                temp.append(float(self.table.item(row, col).text()))
            data.append(temp)

        df = pd.DataFrame()
        for i in range(len(data)):
            df[self.label[index_col[i]]] = data[i]

        for i in reversed(range(self.layout_analysis.count())):
            self.layout_analysis.itemAt(i).widget().setParent(None)

        fig_analysis = sns.pairplot(df, hue=self.combo_box_analysis.currentText()).fig
        canvas_analysis = FigureCanvas(fig_analysis)

        self.layout_analysis.addWidget(canvas_analysis)

    def on_click_clear_regression(self):
        self.list_regression.clear()
        self.list_regression.addItems(self.label[0:])

        self.canvas_regressions.axes.cla()

        self.input_regression_X.clear()
        self.input_regression_Y.clear()

    def on_click_add_x(self):
        try:
            self.list_regression_X.addItems([self.list_regression.selectedItems()[0].text()])
            self.list_regression.takeItem(self.list_regression.currentRow())
        except:
            pass

    def on_click_add_y(self):
        try:
            self.list_regression_Y.addItems([self.list_regression.selectedItems()[0].text()])
            self.list_regression.takeItem(self.list_regression.currentRow())
        except:
            pass

    def on_click_building_regression(self):
        x_index = [self.label.index(i) for i in self.input_regression_X.text().split('; ')]

        X = []
        for col in x_index:
            for row in range(self.table.rowCount()):
                X.append(float(self.table.item(row, col).text()))

        y_index = [self.label.index(i) for i in self.input_regression_Y.text().split('; ')]

        y = []
        for col in y_index:
            for row in range(self.table.rowCount()):
                y.append(float(self.table.item(row, col).text()))

        self.canvas_regressions.axes.cla()
        sns.scatterplot(x=X, y=y, ax=self.canvas_regressions.axes)

        X = np.array(X).reshape((-1, 1))
        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0)

        model = LinearRegression().fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        print('MSE {:.3f}'.format(
            mean_squared_error(y_test, y_test_pred)))
        print('R^2 test: {:.3f}'.format(
            r2_score(y_test, y_test_pred)))

        self.canvas_regressions.axes.plot(X, model.predict(X), color='red', linewidth=2)
        self.canvas_regressions.draw()

        # data = []
        # for i in range(self.ui.tableWidget.rowCount()):
        #     data.append(self.ui.tableWidget.item(i, column).text())

    def on_click_delete_row(self):
        try:
            index = self.table.selectionModel().selectedRows()[0].row()
            for _ in range(len(self.table.selectionModel().selectedRows())):
                try:
                    self.table.removeRow(index)
                except:
                    pass
        except:
            pass

        self.update_label_size()

    def on_click_delete_column(self):
        try:
            index = self.table.selectionModel().selectedColumns()[0].column()
            for _ in range(len(self.table.selectionModel().selectedColumns())):
                try:
                    self.table.removeColumn(index)
                    self.label.pop(index)
                except:
                    pass
        except:
            pass

        self.update_label_size()

    def on_click_clear_nan(self):
        for row in range(self.table.rowCount()):
            for col in range(self.table.columnCount()):
                try:
                    if self.table.item(row, col).text() == "NaN":
                        self.table.removeRow(row)
                        continue
                except:
                    pass

        self.update_label_size()

    def on_click_add_column(self):
        try:
            self.table.insertColumn(int(self.input_column.text()) - 1)
        except:
            pass

        self.input_column.setText("")
        self.update_label_size()

    def on_click_add_row(self):
        try:
            index = self.table.selectionModel().selectedRows()[0].row()
            self.table.insertRow(index + 1)
        except:
            pass

    def update_label_size(self):
        size = "Размер таблицы: " + str(self.table.columnCount()) + 'x' + str(self.table.rowCount())
        self.label_size.setText(size)

    def on_click_clear_table(self):
        self.table.setRowCount(16)
        self.table.setColumnCount(6)

        for row in range(self.table.rowCount()):
            for col in range(self.table.columnCount()):
                self.table.setItem(row, col, QTableWidgetItem(""))

        self.label = ['1', '2', '3', '4', '5', '6']
        self.table.setColumnCount(len(self.label))
        self.table.setHorizontalHeaderLabels(self.label)

        for row in range(self.table.rowCount()):
            self.table.setColumnWidth(row, 150)

        self.update_label_size()

        self.list_regression.clear()

    def openFile(self):
        self.on_click_clear_table()

        fname = QFileDialog.getOpenFileName(self, 'Open file')[0]
        try:
            with open(fname) as file:
                row_count = 0
                for _ in file:
                    row_count += 1
                self.table.setRowCount(row_count - 1)

            with open(fname) as file:
                for i, line in enumerate(file):
                    line = str.replace(line, '\n', '')
                    if i == 0:
                        self.label = line.split(',')
                        self.table.setColumnCount(len(self.label))
                        self.table.setHorizontalHeaderLabels(self.label)
                    else:
                        for j, cell in enumerate(line.split(',')):
                            self.table.setItem(i - 1, j, QTableWidgetItem(cell))

            self.table.resizeColumnsToContents()  # ширина столцов подогнать по ширине текста
        except:
            pass

        self.list_regression.addItems(self.label[0:])
        self.list_classification.addItems(self.label[0:])
        self.list_analysis.addItems(self.label[0:])
        self.list_clusterization.addItems(self.label[0:])
        self.combo_box_analysis.addItems(self.label[0:])
        self.combo_box_table.addItems(self.label[0:])
        self.combo_box_classification_class.addItems(self.label[0:])
        self.combo_box_clusterization_class.addItems(self.label[0:])

        self.update_label_size()

    def saveFile(self):
        data = ','.join(self.label)
        for row in range(self.table.rowCount()):
            tmp = ""
            for col in range(self.table.columnCount()):
                if tmp != "":
                    try:
                        tmp = ",".join((tmp, self.table.item(row, col).text()))
                    except:
                        tmp = ",".join((tmp, 'NaN'))
                else:
                    try:
                        tmp = self.table.item(row, col).text()
                    except:
                        tmp = 'NaN'
            data = "\n".join((data, tmp))

        name = QFileDialog.getExistingDirectory(self, "Выбрать папку", ".")

        try:
            f = open(name + '/file.csv', 'w')
            f.write(data)
        except:
            pass

        self.list_regression.clear()

        self.update_label_size()

if __name__ == "__main__":
    app = QApplication([])                              # Создание объекта класса QApplication
    app.setStyle('Fusion')                              #
    window = MainWindow()                               #
    window.resize(830, 950)                             # Изменение размера главного окна приложения
    window.setStyleSheet("background-color:#D0E1F9")    #
    window.show()                                       # Отображение виджета на экране монитора
    app.exec_()                                         #