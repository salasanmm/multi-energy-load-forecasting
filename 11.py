import win32com.client
from win32com.client import constants


def create_ies_topology():
    try:
        # 1. 初始化 Visio 应用
        visio = win32com.client.Dispatch("Visio.Application")
        visio.Visible = True

        # 创建一个新文档（使用空白横向页面模板）
        doc = visio.Documents.Add("")
        page = visio.ActivePage

        # 设置页面为横向 A4 尺寸
        page.PageSheet.CellsSRC(1, 10, 11).FormulaU = "11.6929 in"  # 宽度
        page.PageSheet.CellsSRC(1, 10, 12).FormulaU = "8.2677 in"  # 高度

        # 2. 定义颜色 RGB 字符串 (匹配您图片中的全新配色)
        COLOR_ELE = "RGB(235, 130, 30)"  # 橙色 - 电
        COLOR_HEAT = "RGB(180, 20, 100)"  # 玫红 - 热
        COLOR_COOL = "RGB(100, 190, 230)"  # 天蓝 - 冷
        COLOR_GAS = "RGB(90, 10, 90)"  # 深紫 - 气

        # 3. 定义所有节点（名称与图中的相对坐标，单位：英寸）
        # 布局逻辑：左边源、中间转换与存储、右边负荷
        nodes_config = {
            # 外部能源 / 输流侧 (左列)
            "Power Grid": (1.5, 6.8),
            "Wind Energy": (1.5, 5.0),
            "Solar Energy": (1.5, 3.2),
            "Gas Market": (1.5, 1.2),

            # 转换与存储 (中列)
            "Power Storage": (5.5, 7.0),
            "Power to Gas": (4.0, 5.0),
            "CCHP": (4.8, 3.2),
            "Gas Boiler": (4.8, 1.2),

            # 储能与调节 (中右列)
            "Heat Storage": (7.0, 4.2),
            "Cool Storage": (8.2, 4.2),
            "Heat Pump": (7.5, 1.8),

            # 建筑负荷堆叠 (右列)
            "Electricity Load": (10.2, 5.8),
            "Heating Load": (10.2, 4.2),
            "Cooling Load": (10.2, 2.6)
        }

        shapes = {}
        # 4. 批量绘制节点占位框
        for name, (x, y) in nodes_config.items():
            # 绘制一个矩形作为占位符
            shape = page.DrawRectangle(x - 0.6, y - 0.4, x + 0.6, y + 0.4)
            shape.Text = name
            # 简单的美化：白底、细灰边、黑字
            shape.CellsSRC(1, 3, 0).FormulaForceU = "THEMEGUARD(RGB(255,255,255))"
            shape.CellsSRC(1, 2, 0).FormulaU = "RGB(150,150,150)"  # LineColor
            shape.CellsSRC(1, 2, 1).FormulaU = "1 pt"  # LineWeight
            shapes[name] = shape

        # 5. 辅助函数：绘制带颜色和箭头的连接线
        def connect_nodes(from_shape, to_shape, color_rgb, weight="2 pt"):
            connector = page.Drop(visio.Documents.Item("BASIC_M.VSSX").Masters.ItemU("Dynamic connector"), 0,
                                  0) if "BASIC_M.VSSX" in [d.Name for d in visio.Documents] else page.DrawLine(0, 0, 1,
                                                                                                               1)

            # 如果没有标准动态连接线母版，直接用普通线连接
            if connector.NameU.startswith("Dynamic connector") or True:
                # 绑定起点和终点
                from_cell = from_shape.CellsSRC(1, 1, 0)  # AlignBottom/Center等
                to_cell = to_shape.CellsSRC(1, 1, 0)
                connector.CellsSRC(1, 1, 0).GlueTo(from_shape.CellsSRC(7, 1, 1))  # BeginX
                connector.CellsSRC(1, 1, 1).GlueTo(to_shape.CellsSRC(7, 2, 1))  # EndX

            # 设置连线颜色、线宽和箭头
            connector.CellsSRC(1, 2, 0).FormulaU = color_rgb  # LineColor
            connector.CellsSRC(1, 2, 1).FormulaU = weight  # LineWeight
            connector.CellsSRC(1, 2, 3).FormulaU = "4"  # LineEndArrow (4代表标准箭头)
            return connector

        # 6. 严格按照图中的拓扑关系连线
        # 电力网络 (橙色)
        connect_nodes(shapes["Power Grid"], shapes["Power Storage"], COLOR_ELE)
        connect_nodes(shapes["Power Grid"], shapes["Power to Gas"], COLOR_ELE)
        connect_nodes(shapes["Wind Energy"], shapes["Power Storage"], COLOR_ELE)
        connect_nodes(shapes["Solar Energy"], shapes["Power Storage"], COLOR_ELE)
        connect_nodes(shapes["Power Storage"], shapes["Electricity Load"], COLOR_ELE)
        connect_nodes(shapes["Power Storage"], shapes["Heat Pump"], COLOR_ELE)
        connect_nodes(shapes["CCHP"], shapes["Power Storage"], COLOR_ELE)

        # 天然气网络 (深紫)
        connect_nodes(shapes["Gas Market"], shapes["CCHP"], COLOR_GAS)
        connect_nodes(shapes["Gas Market"], shapes["Gas Boiler"], COLOR_GAS)
        connect_nodes(shapes["Power to Gas"], shapes["CCHP"], COLOR_GAS)

        # 供热网络 (玫红)
        connect_nodes(shapes["Gas Boiler"], shapes["Heating Load"], COLOR_HEAT)
        connect_nodes(shapes["CCHP"], shapes["Heat Storage"], COLOR_HEAT)
        connect_nodes(shapes["Heat Storage"], shapes["Heating Load"], COLOR_HEAT)
        connect_nodes(shapes["Heat Pump"], shapes["Heat Storage"], COLOR_HEAT)

        # 供冷网络 (天蓝)
        connect_nodes(shapes["CCHP"], shapes["Cool Storage"], COLOR_COOL)
        connect_nodes(shapes["Cool Storage"], shapes["Cooling Load"], COLOR_COOL)
        connect_nodes(shapes["Heat Pump"], shapes["Cool Storage"], COLOR_COOL)

        # 7. 绘制右上角 Legend 外部框
        legend_box = page.DrawRectangle(6.5, 5.5, 8.5, 7.8)
        legend_box.Text = "Legend\n\n   Electricity\n   Heating\n   Cooling\n   Fossil / Gas"
        legend_box.CellsSRC(1, 3, 0).FormulaU = "RGB(245,245,245)"

        print("Visio 拓扑图生成成功！")

    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    create_ies_topology()