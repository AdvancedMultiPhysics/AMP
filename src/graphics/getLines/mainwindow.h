#ifndef TimerWindow_H
#define TimerWindow_H

#include "AMP/graphics/RGBA.h"
#include "AMP/utils/Array.h"

#include "ThreadedSlotsClass.h"

#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QPushButton>
#include <QScrollArea>
#include <QTableWidget>
#include <QTimer>
#include <QToolBar>
#include <QToolButton>

#include <memory>


class QAction;
class QMenu;
class QTableView;
class TraceWindow;
class MemoryWindow;
class DrawBoxClass;


class MainWindow : public ThreadedSlotsClass
{
    Q_OBJECT

public:
    MainWindow();
    MainWindow( const MainWindow & )            = delete;
    MainWindow &operator=( const MainWindow & ) = delete;
    virtual ~MainWindow();

protected:
    void closeEvent( QCloseEvent *event );

public slots:
    void exit();
private slots:
    void close();
    void open();
    void reset();
    void about();

    void resizeEvent( QResizeEvent *e );
    void resizeDone();

    // Developer functions
    void savePerformance();
    void runUnitTestsSlot();

private:
    void createActions();
    void createMenus();
    void createToolBars();
    void createStatusBar();
    void readSettings();
    void writeSettings();
    void loadFile( const std::string &filename, bool showFailure = true );
    void setCurrentFile( const QString &fileName );
    void updateDisplay();
    void mousePressEvent( QMouseEvent *event );
    void mouseMoveEvent( QMouseEvent *event );
    void mouseReleaseEvent( QMouseEvent *event );
    QString strippedName( const QString &fullFileName );

    QMenuBar *mainMenu;

    QMenu *fileMenu;
    QMenu *editMenu;
    QMenu *helpMenu;
    QToolBar *fileToolBar;
    QToolBar *editToolBar;
    QAction *closeAct;
    QAction *openAct;
    QAction *resetAct;
    QAction *exitAct;
    QAction *aboutAct;
    QAction *savePerformanceTimers;
    QAction *runUnitTestAction;

    QTimer resizeTimer;
    std::unique_ptr<QMenu> processorButtonMenu;

    QLabel *imageWindow;

    std::shared_ptr<DrawBoxClass> gbox;

protected: // Internal data
    std::string d_lastPath;
    AMP::Array<AMP::Graphics::ARGB32> d_data;

public: // Data for unit testing
    bool runUnitTests( const std::string &file );

private:
    std::string unitTestFilename;
    void resetUnitTestRunning();
    void callLoadFile();
};

#endif
