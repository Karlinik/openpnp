/*
 * Copyright (C) 2011 Jason von Nieda <jason@vonnieda.org>
 * 
 * This file is part of OpenPnP.
 * 
 * OpenPnP is free software: you can redistribute it and/or modify it under the terms of the GNU
 * General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * OpenPnP is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with OpenPnP. If not, see
 * <http://www.gnu.org/licenses/>.
 * 
 * For more information about OpenPnP visit http://openpnp.org
 */

package org.openpnp.gui;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.event.ActionEvent;

import javax.swing.*;
import javax.swing.border.EtchedBorder;
import javax.swing.border.TitledBorder;

import org.jdesktop.beansbinding.AutoBinding;
import org.jdesktop.beansbinding.AutoBinding.UpdateStrategy;
import org.jdesktop.beansbinding.BeanProperty;
import org.jdesktop.beansbinding.Bindings;
import org.openpnp.gui.components.AutoSelectTextTable;
import org.openpnp.gui.components.CameraView;
import org.openpnp.gui.components.ComponentDecorators;
import org.openpnp.gui.components.reticle.FootprintReticle;
import org.openpnp.gui.components.reticle.Reticle;
import org.openpnp.gui.support.DoubleConverter;
import org.openpnp.gui.support.Helpers;
import org.openpnp.gui.support.Icons;
import org.openpnp.gui.tablemodel.FootprintTableModel;
import org.openpnp.model.*;
import org.openpnp.model.Footprint.Pad;
import org.openpnp.model.Package;
import org.openpnp.spi.Camera;

import com.jgoodies.forms.layout.ColumnSpec;
import com.jgoodies.forms.layout.FormLayout;
import com.jgoodies.forms.layout.FormSpecs;
import com.jgoodies.forms.layout.RowSpec;
import org.openpnp.util.UiUtils;
import org.openpnp.util.VisionUtils;
import org.openpnp.vision.pipeline.CvPipeline;
import org.openpnp.vision.pipeline.ui.CvPipelineEditor;
import org.openpnp.vision.pipeline.ui.CvPipelineEditorDialog;

@SuppressWarnings("serial")
public class PackageVisionPanel extends JPanel {
    private FootprintTableModel tableModel;
    private JTable table;

    private final Footprint footprint;
    private final Package pkg;

    public PackageVisionPanel(Package pkg) {
        this.pkg = pkg;
        this.footprint = pkg.getFootprint();
        initialize();
    }

    private void initialize() {
        setLayout(new BorderLayout(0, 0));
        tableModel = new FootprintTableModel(footprint);

        deleteAction.setEnabled(false);

        JPanel propertiesPanel = new JPanel();
        add(propertiesPanel, BorderLayout.NORTH);
        propertiesPanel.setBorder(
                new TitledBorder(new EtchedBorder(EtchedBorder.LOWERED, null, null), "Settings",
                        TitledBorder.LEADING, TitledBorder.TOP, null));
        propertiesPanel.setLayout(new FormLayout(
                new ColumnSpec[] {FormSpecs.RELATED_GAP_COLSPEC, FormSpecs.DEFAULT_COLSPEC,
                        FormSpecs.RELATED_GAP_COLSPEC, ColumnSpec.decode("default:grow"),},
                new RowSpec[] {FormSpecs.RELATED_GAP_ROWSPEC, FormSpecs.DEFAULT_ROWSPEC,
                        FormSpecs.RELATED_GAP_ROWSPEC, FormSpecs.DEFAULT_ROWSPEC,
                        FormSpecs.RELATED_GAP_ROWSPEC, FormSpecs.DEFAULT_ROWSPEC,}));

        JLabel lblUnits = new JLabel("Units");
        propertiesPanel.add(lblUnits, "2, 2, right, default");

        unitsCombo = new JComboBox(LengthUnit.values());
        propertiesPanel.add(unitsCombo, "4, 2, left, default");

        JLabel lblBodyWidth = new JLabel("Body Width");
        propertiesPanel.add(lblBodyWidth, "2, 4, right, default");

        bodyWidthTf = new JTextField();
        propertiesPanel.add(bodyWidthTf, "4, 4, left, default");
        bodyWidthTf.setColumns(10);

        JLabel lblBodyHeight = new JLabel("Body Length");
        propertiesPanel.add(lblBodyHeight, "2, 6, right, default");

        bodyHeightTf = new JTextField();
        propertiesPanel.add(bodyHeightTf, "4, 6, left, default");
        bodyHeightTf.setColumns(10);

        /////////////////////////////////////////////

        JPanel tablePanel = new JPanel();
        add(tablePanel, BorderLayout.CENTER);
        tablePanel.setBorder(
                new TitledBorder(null, "Pads", TitledBorder.LEADING, TitledBorder.TOP, null, null));

        table = new AutoSelectTextTable(tableModel);
        table.setAutoCreateRowSorter(true);
        table.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        table.getSelectionModel().addListSelectionListener(e -> {
            if (e.getValueIsAdjusting()) {
                return;
            }

            Pad pad = getSelectedPad();

            deleteAction.setEnabled(pad != null);
        });
        tablePanel.setLayout(new BorderLayout(0, 0));

        JPanel toolbarPanel = new JPanel();
        tablePanel.add(toolbarPanel, BorderLayout.NORTH);
        toolbarPanel.setLayout(new BorderLayout(0, 0));

        JToolBar toolBar = new JToolBar();
        toolBar.setFloatable(false);
        toolbarPanel.add(toolBar);

        toolBar.add(newAction);
        toolBar.add(deleteAction);

        JScrollPane tableScrollPane = new JScrollPane(table);
        tableScrollPane.setPreferredSize(new Dimension(454, 100));
        tablePanel.add(tableScrollPane);

        //////////////////////////////////////////////

        JPanel bottomVisionPanel = new JPanel();
        add(bottomVisionPanel, BorderLayout.SOUTH);
        bottomVisionPanel.setBorder(
                new TitledBorder(new EtchedBorder(EtchedBorder.LOWERED, null, null), "Bottom Vision",
                        TitledBorder.LEADING, TitledBorder.TOP, null));
        bottomVisionPanel.setLayout(new FormLayout(
                new ColumnSpec[] {FormSpecs.RELATED_GAP_COLSPEC, FormSpecs.DEFAULT_COLSPEC,
                        FormSpecs.RELATED_GAP_COLSPEC, ColumnSpec.decode("default:grow"),},
                new RowSpec[] {FormSpecs.RELATED_GAP_ROWSPEC, FormSpecs.DEFAULT_ROWSPEC,
                        FormSpecs.RELATED_GAP_ROWSPEC, FormSpecs.DEFAULT_ROWSPEC,
                        FormSpecs.RELATED_GAP_ROWSPEC, FormSpecs.DEFAULT_ROWSPEC,}));

        JLabel lblPipeline = new JLabel("Pipeline");
        JButton editPipelineBtn = new JButton("Edit");
        editPipelineBtn.addActionListener(e -> UiUtils.messageBoxOnException(this::editPipeline));
        JButton resetPipelineBtn = new JButton("Reset to Default");
        resetPipelineBtn.addActionListener((e) -> {
            int result = JOptionPane.showConfirmDialog(getTopLevelAncestor(),
                    "This will replace the current package's and all its parts' pipeline with the default pipeline. Are you sure?", null,
                    JOptionPane.YES_NO_OPTION, JOptionPane.WARNING_MESSAGE);
            if (result == JOptionPane.YES_OPTION) {
                UiUtils.messageBoxOnException(() -> {
                    pkg.setPipeline(Configuration.get().getDefaultPipeline());
                    editPipeline();
                });
            }
        });

        bottomVisionPanel.add(lblPipeline, "2, 2, right, default");
        bottomVisionPanel.add(editPipelineBtn, "4, 2, left, default");
        bottomVisionPanel.add(resetPipelineBtn, "4, 4, left, default");

        showReticle();
        initDataBindings();
    }

    private void editPipeline() throws Exception {
        CvPipeline pipeline = pkg.getCvPipeline();
        pipeline.setProperty("camera", VisionUtils.getBottomVisionCamera());
        pipeline.setProperty("nozzle", MainFrame.get().getMachineControls().getSelectedNozzle());

        CvPipelineEditor editor = new CvPipelineEditor(pipeline);
        JDialog dialog = new CvPipelineEditorDialog(MainFrame.get(), "Bottom Vision Pipeline", editor);
        dialog.setVisible(true);
    }

    protected void initDataBindings() {
        DoubleConverter doubleConverter =
                new DoubleConverter(Configuration.get().getLengthDisplayFormat());

        BeanProperty<Footprint, LengthUnit> footprintBeanProperty = BeanProperty.create("units");
        BeanProperty<JComboBox, Object> jComboBoxBeanProperty = BeanProperty.create("selectedItem");
        AutoBinding<Footprint, LengthUnit, JComboBox, Object> autoBinding =
                Bindings.createAutoBinding(UpdateStrategy.READ_WRITE, footprint,
                        footprintBeanProperty, unitsCombo, jComboBoxBeanProperty);
        autoBinding.bind();
        //
        BeanProperty<Footprint, Double> footprintBeanPropertyWidth = BeanProperty.create("bodyWidth");
        BeanProperty<JTextField, String> jTextFieldBeanProperty = BeanProperty.create("text");
        AutoBinding<Footprint, Double, JTextField, String> autoBindingWidth =
                Bindings.createAutoBinding(UpdateStrategy.READ_WRITE, footprint,
                        footprintBeanPropertyWidth, bodyWidthTf, jTextFieldBeanProperty);
        autoBindingWidth.setConverter(doubleConverter);
        autoBindingWidth.bind();
        //
        BeanProperty<Footprint, Double> footprintBeanPropertyHeight = BeanProperty.create("bodyHeight");
        BeanProperty<JTextField, String> jTextFieldBeanProperty_1 = BeanProperty.create("text");
        AutoBinding<Footprint, Double, JTextField, String> autoBindingHeight =
                Bindings.createAutoBinding(UpdateStrategy.READ_WRITE, footprint,
                        footprintBeanPropertyHeight, bodyHeightTf, jTextFieldBeanProperty_1);
        autoBindingHeight.setConverter(doubleConverter);
        autoBindingHeight.bind();

        ComponentDecorators.decorateWithAutoSelect(bodyWidthTf);
        ComponentDecorators.decorateWithAutoSelect(bodyHeightTf);
    }

    private void showReticle() {
        try {
            Camera camera = Configuration.get().getMachine().getDefaultHead().getDefaultCamera();
            CameraView cameraView = MainFrame.get().getCameraViews().getCameraView(camera);
            if (cameraView == null) {
                return;
            }
            cameraView.removeReticle(PackageVisionPanel.class.getName());
            Reticle reticle = new FootprintReticle(footprint);
            cameraView.setReticle(PackageVisionPanel.class.getName(), reticle);
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    private Pad getSelectedPad() {
        int index = table.getSelectedRow();
        if (index == -1) {
            return null;
        }
        index = table.convertRowIndexToModel(index);
        return tableModel.getPad(index);
    }

    public final Action newAction = new AbstractAction() {
        {
            putValue(SMALL_ICON, Icons.add);
            putValue(NAME, "New Pad...");
            putValue(SHORT_DESCRIPTION, "Create a new pad, specifying it's ID.");
        }

        @Override
        public void actionPerformed(ActionEvent arg0) {
            String name;
            if ((name = JOptionPane.showInputDialog(getTopLevelAncestor(),
                    "Please enter a name for the new pad.")) != null) {
                Pad pad = new Pad();
                pad.setName(name);
                footprint.addPad(pad);
                tableModel.fireTableDataChanged();
                Helpers.selectLastTableRow(table);
            }
        }
    };

    public final Action deleteAction = new AbstractAction() {
        {
            putValue(SMALL_ICON, Icons.delete);
            putValue(NAME, "Delete Pad");
            putValue(SHORT_DESCRIPTION, "Delete the currently selected pad.");
        }

        @Override
        public void actionPerformed(ActionEvent arg0) {
            int ret = JOptionPane.showConfirmDialog(getTopLevelAncestor(),
                    "Are you sure you want to delete " + getSelectedPad().getName() + "?",
                    "Delete " + getSelectedPad().getName() + "?", JOptionPane.YES_NO_OPTION);
            if (ret == JOptionPane.YES_OPTION) {
                footprint.removePad(getSelectedPad());
            }
        }
    };
    private JTextField bodyWidthTf;
    private JTextField bodyHeightTf;
    private JComboBox unitsCombo;
}
