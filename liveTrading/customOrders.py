from ibapi.order import Order


def bracketSellOrder(quantity=1, profitTarget=1.02, stopLoss=0.99, orderId=1):
    # Create Parent Order / Initial Entry
    parent = Order()
    parent.orderId = orderId
    parent.orderType = "MTK"
    parent.action = "BUY"
    parent.totalQuantity = quantity
    parent.transmit = False
    parent.eTradeOnly = False
    parent.firmQuoteOnly = False
    # Profit Target Order
    profitTargetOrder = Order()
    profitTargetOrder.orderId = orderId + 1
    profitTargetOrder.orderType = "LMT"
    profitTargetOrder.action = "SELL"
    profitTargetOrder.totalQuantity = quantity
    profitTargetOrder.lmtPrice = round(profitTarget, 2)
    profitTargetOrder.transmit = True
    profitTargetOrder.eTradeOnly = False
    profitTargetOrder.firmQuoteOnly = False
    # Stop Loss Order
    stopLossOrder = Order()
    stopLossOrder.orderId = orderId + 2
    stopLossOrder.orderType = "STP"
    stopLossOrder.action = "SELL"
    stopLossOrder.totalQuantity = quantity
    stopLossOrder.auxPrice = round(stopLoss, 2)
    stopLossOrder.transmit = True
    stopLossOrder.eTradeOnly = False
    stopLossOrder.firmQuoteOnly = False
    # Bracket Orders Array
    bracketOrders = [parent, profitTargetOrder, stopLossOrder]
    return bracketOrders


def bracketBuyOrder(quantity=1, profitTarget=1.02, stopLoss=0.99, orderId=1):
    # Create Parent Order / Initial Entry
    parent = Order()
    parent.orderId = orderId
    parent.orderType = "MTK"
    parent.action = "SELL"
    parent.totalQuantity = quantity
    parent.transmit = False
    parent.eTradeOnly = False
    parent.firmQuoteOnly = False
    # Profit Target Order
    profitTargetOrder = Order()
    profitTargetOrder.orderId = orderId + 1
    profitTargetOrder.orderType = "LMT"
    profitTargetOrder.action = "BUY"
    profitTargetOrder.totalQuantity = quantity
    profitTargetOrder.lmtPrice = round(profitTarget, 2)
    profitTargetOrder.transmit = True
    profitTargetOrder.eTradeOnly = False
    profitTargetOrder.firmQuoteOnly = False
    # Stop Loss Order
    stopLossOrder = Order()
    stopLossOrder.orderId = orderId + 2
    stopLossOrder.orderType = "STP"
    stopLossOrder.action = "BUY"
    stopLossOrder.totalQuantity = quantity
    stopLossOrder.auxPrice = round(stopLoss, 2)
    stopLossOrder.transmit = True
    stopLossOrder.eTradeOnly = False
    stopLossOrder.firmQuoteOnly = False
    # Bracket Orders Array
    bracketOrders = [parent, profitTargetOrder, stopLossOrder]
    return bracketOrders


def marketBuyOrder(orderId, quantity=1):
    # Create order object
    order = Order()
    order.orderId = orderId
    order.orderType = "MKT"  # or LMT etc..
    order.action = "BUY"  # or "SELL"
    order.totalQuantity = quantity
    order.eTradeOnly = False
    order.firmQuoteOnly = False
    return order


def marketSellOrder(orderId, quantity=1):
    # Create order object
    order = Order()
    order.orderId = orderId
    order.orderType = "MKT"
    order.action = "SELL"
    order.totalQuantity = quantity
    order.eTradeOnly = False
    order.firmQuoteOnly = False
    return order
