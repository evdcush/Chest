// Write collection path into the "Court" field for selected items
// Path format: "Top/Child/Subchild".
// If an item is in multiple collections, use the first one.

async function getCollectionPath(collectionID) {
    const collection = await Zotero.Collections.getAsync(collectionID);
    if (!collection) return null;

    let names = [collection.name];
    let parentID = collection.parentID;

    while (parentID) {
        const parent = await Zotero.Collections.getAsync(parentID);
        if (!parent) break;
        names.unshift(parent.name);
        parentID = parent.parentID;
    }

    return names.join("/");
}

async function main() {
    const ZoteroPane = Zotero.getActiveZoteroPane();
    const items = ZoteroPane.getSelectedItems();

    if (!items.length) {
        return "No items selected.";
    }

    // Internal field name for the Court field
    // "court" is NOT DEFINED FOR MOST SHIT
    // we have to exploit a different useless value for our hack
    // "Location in Archive" is defined for preprints, and NO ITEMS
    // in my entire collection have it defined; let's try that.
    // NB: "Loc. in Archive" key is observed as "coverage" in an RDF file.
    //const fieldName = "court";
    //const fieldName = "coverage";
    const fieldName = "archiveLocation";
    // UNSUPPORTED TYPES for "archiveLocation"
    // i just fucking manually selected around these types:
    //   - forumPost
    //   - BlogPost
    //   - WebPage
    //   - Presentation
    const fieldID = Zotero.ItemFields.getID(fieldName);

    await Zotero.DB.executeTransaction(async function () {
        for (let item of items) {
            // Only regular items (skip notes, attachments, etc.)
            if (!item.isRegularItem()) continue;

            const collections = item.getCollections();
            let newValue = "";

            if (collections && collections.length > 0) {
                const primaryCollectionID = collections[0];
                const path = await getCollectionPath(primaryCollectionID);
                if (path) newValue = path;
            }

            // Court is not mapped for most item types, so use the base fieldID
            item.setField(fieldID, newValue);
            await item.save();
        }
    });

    return items.length + " item(s) updated.";
}

await main();
