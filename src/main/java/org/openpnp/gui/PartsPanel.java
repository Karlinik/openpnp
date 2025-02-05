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
import java.awt.Frame;
import java.awt.Toolkit;
import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.DataFlavor;
import java.awt.datatransfer.StringSelection;
import java.awt.event.ActionEvent;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.prefs.Preferences;
import java.util.regex.PatternSyntaxException;
import java.util.stream.Collectors;

import javax.swing.AbstractAction;
import javax.swing.Action;
import javax.swing.DefaultCellEditor;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSplitPane;
import javax.swing.JTabbedPane;
import javax.swing.JTable;
import javax.swing.JTextField;
import javax.swing.JToolBar;
import javax.swing.ListSelectionModel;
import javax.swing.RowFilter;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import javax.swing.table.TableRowSorter;

import org.openpnp.gui.components.AutoSelectTextTable;
import org.openpnp.gui.support.*;
import org.openpnp.gui.tablemodel.PartsTableModel;
import org.openpnp.model.Configuration;
import org.openpnp.model.Package;
import org.openpnp.model.Part;
import org.openpnp.model.Pipeline;
import org.openpnp.spi.Feeder;
import org.openpnp.spi.FiducialLocator;
import org.openpnp.util.UiUtils;
import org.pmw.tinylog.Logger;
import org.simpleframework.xml.Serializer;

import static javax.swing.SwingConstants.TOP;

@SuppressWarnings("serial")
public class PartsPanel extends JPanel implements WizardContainer {


    private static final String PREF_DIVIDER_POSITION = "PartsPanel.dividerPosition";
    private static final int PREF_DIVIDER_POSITION_DEF = -1;
    private Preferences prefs = Preferences.userNodeForPackage(PartsPanel.class);

    private final Configuration configuration;
    private final Frame frame;

    private PartsTableModel tableModel;
    private TableRowSorter<PartsTableModel> tableSorter;
    private JTextField searchTextField;
    private JTable table;
    private ActionGroup singleSelectionActionGroup;
    private ActionGroup multiSelectionActionGroup;
    private JTabbedPane tabbedPane;

    public PartsPanel(Configuration configuration, Frame frame) {
        this.configuration = configuration;
        this.frame = frame;

        singleSelectionActionGroup = new ActionGroup(deletePartAction, pickPartAction, copyPartToClipboardAction);
        singleSelectionActionGroup.setEnabled(false);
        multiSelectionActionGroup = new ActionGroup(deletePartAction);
        multiSelectionActionGroup.setEnabled(false);

        setLayout(new BorderLayout(0, 0));

        createAndAddToolbar();

        tableModel = new PartsTableModel();
        tableSorter = new TableRowSorter<>(tableModel);

        JSplitPane splitPane = new JSplitPane();
        splitPane.setOrientation(JSplitPane.VERTICAL_SPLIT);
        splitPane.setContinuousLayout(true);
        splitPane
                .setDividerLocation(prefs.getInt(PREF_DIVIDER_POSITION, PREF_DIVIDER_POSITION_DEF));
        splitPane.addPropertyChangeListener("dividerLocation",
                evt -> prefs.putInt(PREF_DIVIDER_POSITION, splitPane.getDividerLocation()));
        add(splitPane, BorderLayout.CENTER);

        tabbedPane = new JTabbedPane(TOP);

        tableSetup();

        splitPane.setLeftComponent(new JScrollPane(table));
        splitPane.setRightComponent(tabbedPane);
    }

    private void createAndAddToolbar() {
        JPanel toolbarAndSearch = new JPanel();
        add(toolbarAndSearch, BorderLayout.NORTH);
        toolbarAndSearch.setLayout(new BorderLayout(0, 0));

        JToolBar toolBar = new JToolBar();
        toolBar.setFloatable(false);
        toolbarAndSearch.add(toolBar);

        JPanel upperPanel = new JPanel();
        toolbarAndSearch.add(upperPanel, BorderLayout.EAST);

        JLabel lblSearch = new JLabel("Search");
        upperPanel.add(lblSearch);

        searchTextField = new JTextField();
        searchTextField.getDocument().addDocumentListener(new DocumentListener() {
            @Override
            public void removeUpdate(DocumentEvent e) {
                search();
            }

            @Override
            public void insertUpdate(DocumentEvent e) {
                search();
            }

            @Override
            public void changedUpdate(DocumentEvent e) {
                search();
            }
        });
        upperPanel.add(searchTextField);
        searchTextField.setColumns(15);

        toolBar.add(newPartAction);
        toolBar.add(deletePartAction);
        toolBar.addSeparator();
        toolBar.add(pickPartAction);

        toolBar.addSeparator();
        JButton copyToClipboardButton = new JButton(copyPartToClipboardAction);
        copyToClipboardButton.setHideActionText(true);
        toolBar.add(copyToClipboardButton);

        JButton pasteFromClipboardButton = new JButton(pastePartFromClipboardAction);
        pasteFromClipboardButton.setHideActionText(true);
        toolBar.add(pasteFromClipboardButton);
    }

    private void tableSetup() {
        JComboBox<Package> packagesCombo = new JComboBox<>(new PackagesComboBoxModel());
        packagesCombo.setMaximumRowCount(20);
        packagesCombo.setRenderer(new IdentifiableListCellRenderer<>());

        JComboBox<Pipeline> pipelinesCombo = new JComboBox<>(new PipelinesComboBoxModel());
        pipelinesCombo.setMaximumRowCount(20);
        pipelinesCombo.setRenderer(new IdentifiableListCellRenderer<>());

        table = new AutoSelectTextTable(tableModel);
        table.setSelectionMode(ListSelectionModel.MULTIPLE_INTERVAL_SELECTION);
        table.setDefaultEditor(Package.class,
                new DefaultCellEditor(packagesCombo));
        table.setDefaultRenderer(Package.class,
                new IdentifiableTableCellRenderer<Package>());

        table.setDefaultEditor(Pipeline.class,
                new DefaultCellEditor(pipelinesCombo));
        table.setDefaultRenderer(Pipeline.class,
                new IdentifiableTableCellRenderer<Pipeline>());

        table.setRowSorter(tableSorter);
        table.getTableHeader().setDefaultRenderer(new MultisortTableHeaderCellRenderer());

        table.getSelectionModel().addListSelectionListener(e -> {
            if (e.getValueIsAdjusting()) {
                return;
            }

            List<Part> selections = getSelections();

            if (selections.size() > 1) {
                singleSelectionActionGroup.setEnabled(false);
                multiSelectionActionGroup.setEnabled(true);
            } else {
                multiSelectionActionGroup.setEnabled(false);
                singleSelectionActionGroup.setEnabled(!selections.isEmpty());
            }

            Part part = getSelection();

            int selectedTab = tabbedPane.getSelectedIndex();
            tabbedPane.removeAll();

            if (part != null) {
                partSelectionSetup(part);
            }

            if (selectedTab >= 0 && selectedTab < tabbedPane.getTabCount()) {
                tabbedPane.setSelectedIndex(selectedTab);
            }

            revalidate();
            repaint();
        });
    }

    private void partSelectionSetup(Part part) {
        tabbedPane.add("Settings", new JScrollPane(new PartSettingsPanel(part)));

        createBottomVisionPanel(part);
        createFiducialLocatorPanel(part);
    }

    private void createBottomVisionPanel(Part part) {
        Configuration.get().getMachine().getPartAlignments().forEach(partAlignment ->
            addPanelToTabbedPane(partAlignment.getPartConfigurationWizard(part)));
    }

    private void createFiducialLocatorPanel(Part part) {
        FiducialLocator fiducialLocator = Configuration.get().getMachine().getFiducialLocator();
        addPanelToTabbedPane(fiducialLocator.getPartConfigurationWizard(part));
    }

    private void addPanelToTabbedPane(Wizard wizard) {
        if (wizard != null) {
            JPanel panel = new JPanel();
            panel.setLayout(new BorderLayout());
            panel.add(wizard.getWizardPanel());
            tabbedPane.add(wizard.getWizardName(), new JScrollPane(panel));
            wizard.setWizardContainer(PartsPanel.this);
        }
    }

    private Part getSelection() {
        List<Part> selections = getSelections();
        if (selections.size() != 1) {
            return null;
        }
        return selections.get(0);
    }

    private List<Part> getSelections() {
        List<Part> selections = new ArrayList<>();
        for (int selectedRow : table.getSelectedRows()) {
            selectedRow = table.convertRowIndexToModel(selectedRow);
            selections.add(tableModel.getPart(selectedRow));
        }
        return selections;
    }

    private void search() {
        RowFilter<PartsTableModel, Object> rf = null;
        // If current expression doesn't parse, don't update.
        try {
            rf = RowFilter.regexFilter("(?i)" + searchTextField.getText().trim());
        } catch (PatternSyntaxException e) {
            Logger.warn(e, "Search failed");
            return;
        }
        tableSorter.setRowFilter(rf);
    }

    public final Action newPartAction = new AbstractAction() {
        {
            putValue(SMALL_ICON, Icons.add);
            putValue(NAME, "New Part...");
            putValue(SHORT_DESCRIPTION, "Create a new part, specifying it's ID.");
        }

        @Override
        public void actionPerformed(ActionEvent arg0) {
            if (Configuration.get().getPackages().isEmpty()) {
                MessageBoxes.errorBox(getTopLevelAncestor(), "Error",
                        "There are currently no packages defined in the system. Please create at least one package before creating a part.");
                return;
            }

            String id;
            while ((id = JOptionPane.showInputDialog(frame,
                    "Please enter an ID for the new part.")) != null) {
                if (configuration.getPart(id) == null) {
                    Part part = new Part(id);

                    part.setPackage(Configuration.get().getPackages().get(0));

                    configuration.addPart(part);
                    tableModel.fireTableDataChanged();
                    Helpers.selectLastTableRow(table);
                    break;
                }

                MessageBoxes.errorBox(frame, "Error", "Part ID " + id + " already exists.");
            }
        }
    };

    public final Action deletePartAction = new AbstractAction() {
        {
            putValue(SMALL_ICON, Icons.delete);
            putValue(NAME, "Delete Part");
            putValue(SHORT_DESCRIPTION, "Delete the currently selected part.");
        }

        @Override
        public void actionPerformed(ActionEvent arg0) {
            List<Part> selections = getSelections();
            List<String> ids = selections.stream().map(Part::getId).collect(Collectors.toList());
            String formattedIds;
            if (ids.size() <= 3) {
                formattedIds = String.join(", ", ids);
            } else {
                formattedIds = String.join(", ", ids.subList(0, 3)) + ", and " + (ids.size() - 3) + " others";
            }

            int ret = JOptionPane.showConfirmDialog(getTopLevelAncestor(),
                    "Are you sure you want to delete " + formattedIds + "?",
                    "Delete " + selections.size() + " parts?", JOptionPane.YES_NO_OPTION);
            if (ret == JOptionPane.YES_OPTION) {
                for (Part part : selections) {
                    Configuration.get().removePart(part);
                }
            }
        }
    };

    public final Action pickPartAction = new AbstractAction() {
        {
            putValue(SMALL_ICON, Icons.pick);
            putValue(NAME, "Pick Part");
            putValue(SHORT_DESCRIPTION, "Pick the selected part from the first available feeder.");
        }

        @Override
        public void actionPerformed(ActionEvent arg0) {
            UiUtils.submitUiMachineTask(() -> {
                Part part = getSelection();
                Feeder feeder = null;
                // find a feeder to feed
                for (Feeder f : Configuration.get().getMachine().getFeeders()) {
                    if (f.getPart() == part && f.isEnabled()) {
                        feeder = f;
                    }
                }
                if (feeder == null) {
                    throw new Exception("No valid feeder found for " + part.getId());
                }
                // Perform the whole Job like pick cycle as in the FeedersPanel. 
                FeedersPanel.pickFeeder(feeder);
            });
        }
    };

    public final Action copyPartToClipboardAction = new AbstractAction() {
        {
            putValue(SMALL_ICON, Icons.copy);
            putValue(NAME, "Copy Part to Clipboard");
            putValue(SHORT_DESCRIPTION,
                    "Copy the currently selected part to the clipboard in text format.");
        }

        @Override
        public void actionPerformed(ActionEvent arg0) {
            Part part = getSelection();
            if (part == null) {
                return;
            }
            try {
                Serializer s = Configuration.createSerializer();
                StringWriter w = new StringWriter();
                s.write(part, w);
                StringSelection stringSelection = new StringSelection(w.toString());
                Clipboard clipboard = Toolkit.getDefaultToolkit().getSystemClipboard();
                clipboard.setContents(stringSelection, null);
            } catch (Exception e) {
                MessageBoxes.errorBox(getTopLevelAncestor(), "Copy Failed", e);
            }
        }
    };

    public final Action pastePartFromClipboardAction = new AbstractAction() {
        {
            putValue(SMALL_ICON, Icons.paste);
            putValue(NAME, "Create Part from Clipboard");
            putValue(SHORT_DESCRIPTION, "Create a new part from a definition on the clipboard.");
        }

        @Override
        public void actionPerformed(ActionEvent arg0) {
            try {
                Serializer ser = Configuration.createSerializer();
                Clipboard clipboard = Toolkit.getDefaultToolkit().getSystemClipboard();
                String s = (String) clipboard.getData(DataFlavor.stringFlavor);
                Part part = ser.read(Part.class, s);
                for (int i = 0; ; i++) {
                    if (Configuration.get().getPart(part.getId() + "-" + i) == null) {
                        part.setId(part.getId() + "-" + i);
                        Configuration.get().addPart(part);
                        break;
                    }
                }
                tableModel.fireTableDataChanged();
                Helpers.selectLastTableRow(table);
            } catch (Exception e) {
                MessageBoxes.errorBox(getTopLevelAncestor(), "Paste Failed", e);
            }
        }
    };

    @Override
    public void wizardCompleted(Wizard wizard) {
    }

    @Override
    public void wizardCancelled(Wizard wizard) {
    }
}
